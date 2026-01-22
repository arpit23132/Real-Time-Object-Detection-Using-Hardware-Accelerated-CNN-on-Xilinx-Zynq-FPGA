#include <iostream>
#include <chrono>
#include <cstring> 
#include <string>
#include "data.h"  

using namespace std;

/* ================== DIMENSIONS ================== */
#define IMG_H 32
#define IMG_W 32
#define IMG_C 3
#define KERNEL 3

#define CONV1_FILTERS 16
#define CONV2_FILTERS 32

#define POOL1_OUT 16
#define POOL2_OUT 8

#define FC_IN (CONV2_FILTERS * POOL2_OUT * POOL2_OUT) // 2048
#define FC_OUT 10

/* ================== CNN LAYERS ================== */

// Convolution Layer 1 with Padding=1 and Integrated ReLU
void conv2d_layer1(float input[IMG_C][IMG_H][IMG_W], float kernels[CONV1_FILTERS][IMG_C][KERNEL][KERNEL], float bias[CONV1_FILTERS], float output[CONV1_FILTERS][IMG_H][IMG_W]) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < IMG_H; i++) {
            for (int j = 0; j < IMG_W; j++) {
                float sum = 0;
                for (int c = 0; c < IMG_C; c++) {
                    for (int ki = 0; ki < KERNEL; ki++) {
                        for (int kj = 0; kj < KERNEL; kj++) {
                            int in_i = i + ki - 1;
                            int in_j = j + kj - 1;
                            if (in_i >= 0 && in_i < IMG_H && in_j >= 0 && in_j < IMG_W) {
                                sum += input[c][in_i][in_j] * kernels[f][c][ki][kj];
                            }
                        }
                    }
                }
                float val = sum + bias[f];
                output[f][i][j] = (val > 0) ? val : 0; 
            }
        }
    }
}

// Unified MaxPool function using pointers to avoid "incompatible type" errors
void maxpool_generic(int filters, int in_size, float* input, float* output) {
    int out_size = in_size / 2;
    for (int f = 0; f < filters; f++) {
        for (int i = 0; i < out_size; i++) {
            for (int j = 0; j < out_size; j++) {
                int base_in = f * in_size * in_size;
                int base_out = f * out_size * out_size;
                
                float max_val = input[base_in + (i*2)*in_size + (j*2)];
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        float val = input[base_in + (i*2 + pi)*in_size + (j*2 + pj)];
                        if (val > max_val) max_val = val;
                    }
                }
                output[base_out + i*out_size + j] = max_val;
            }
        }
    }
}

// Convolution Layer 2 with Padding=1 and Integrated ReLU
void conv2d_layer2(float input[CONV1_FILTERS][POOL1_OUT][POOL1_OUT], float kernels[CONV2_FILTERS][CONV1_FILTERS][KERNEL][KERNEL], float bias[CONV2_FILTERS], float output[CONV2_FILTERS][POOL1_OUT][POOL1_OUT]) {
    for (int f_out = 0; f_out < CONV2_FILTERS; f_out++) {
        for (int i = 0; i < POOL1_OUT; i++) {
            for (int j = 0; j < POOL1_OUT; j++) {
                float sum = 0;
                for (int f_in = 0; f_in < CONV1_FILTERS; f_in++) {
                    for (int ki = 0; ki < KERNEL; ki++) {
                        for (int kj = 0; kj < KERNEL; kj++) {
                            int in_i = i + ki - 1;
                            int in_j = j + kj - 1;
                            if (in_i >= 0 && in_i < POOL1_OUT && in_j >= 0 && in_j < POOL1_OUT) {
                                sum += input[f_in][in_i][in_j] * kernels[f_out][f_in][ki][kj];
                            }
                        }
                    }
                }
                float val = sum + bias[f_out];
                output[f_out][i][j] = (val > 0) ? val : 0;
            }
        }
    }
}

void fully_connected(float input[FC_IN], float weights[FC_OUT][FC_IN], float bias[FC_OUT], float output[FC_OUT]) {
    for (int i = 0; i < FC_OUT; i++) {
        float sum = 0;
        for (int j = 0; j < FC_IN; j++) sum += input[j] * weights[i][j];
        output[i] = sum + bias[i];
    }
}

/* ================== TOP INFERENCE FUNCTION ================== */

int cnn_inference(float image[IMG_C][IMG_H][IMG_W], 
                  float w1[16][3][3][3], float b1[16], 
                  float w2[32][16][3][3], float b2[32], 
                  float wfc[10][2048], float bfc[10]) {
    
    static float out1[16][32][32], p1[16][16][16];
    static float out2[32][16][16], p2[32][8][8];
    static float flat[2048], fc_out[10];

    // Conv 1 -> Pool 1
    conv2d_layer1(image, w1, b1, out1);
    maxpool_generic(16, 32, (float*)out1, (float*)p1);

    // Conv 2 -> Pool 2
    conv2d_layer2(p1, w2, b2, out2);
    maxpool_generic(32, 16, (float*)out2, (float*)p2);

    // Flattening (PyTorch order: Channels, Height, Width)
    int idx = 0;
    for(int f=0; f<32; f++) 
        for(int i=0; i<8; i++) 
            for(int j=0; j<8; j++) 
                flat[idx++] = p2[f][i][j];

    fully_connected(flat, wfc, bfc, fc_out);

    int pred = 0; float max_v = fc_out[0];
    for(int i=1; i<10; i++) {
        if(fc_out[i] > max_v) { 
            max_v = fc_out[i]; 
            pred = i; 
        }
    }
    return pred;
}

/* ================== MAIN ================== */

const float* get_img_ptr(int label) {
    switch(label) {
        case 0: return img0_raw; case 1: return img1_raw; case 2: return img2_raw;
        case 3: return img3_raw; case 4: return img4_raw; case 5: return img5_raw;
        case 6: return img6_raw; case 7: return img7_raw; case 8: return img8_raw;
        case 9: return img9_raw; default: return img0_raw;
    }
}

int main() {
    static float image[IMG_C][IMG_H][IMG_W];
    static float w1[16][3][3][3], b1[16];
    static float w2[32][16][3][3], b2[32];
    static float wfc[10][2048], bfc[10];

    // Map weights from data.h
    memcpy(w1, w1_raw, sizeof(w1));
    memcpy(b1, b1_raw, sizeof(b1));
    memcpy(w2, w2_raw, sizeof(w2));
    memcpy(b2, b2_raw, sizeof(b2));
    memcpy(wfc, wfc_raw, sizeof(wfc));
    memcpy(bfc, bfc_raw, sizeof(bfc));

    cout << "=== CIFAR-10 ARM PS Baseline (Fixed Dimensions) ===" << endl;
    const string classes[10] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

    double total_ms = 0;
    int passed = 0;

    for (int i = 0; i < 10; i++) {
        memcpy(image, get_img_ptr(i), sizeof(image));

        auto start = chrono::high_resolution_clock::now();
        int pred = cnn_inference(image, w1, b1, w2, b2, wfc, bfc);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> dur = end - start;
        total_ms += dur.count();

        cout << "Label: " << classes[i] << " | Pred: " << classes[pred];
        if(pred == i) { cout << " [OK]"; passed++; } else { cout << " [FAIL]"; }
        cout << " | " << dur.count() << " ms" << endl;
    }

    cout << "\n----------------------------------------" << endl;
    cout << "Accuracy: " << passed << "/10" << endl;
    cout << "Avg Latency: " << total_ms/10.0 << " ms" << endl;
    cout << "----------------------------------------" << endl;

    return 0;
}