#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;

/* ================== DIMENSIONS ================== */           

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3        // CIFAR-10 has 3 color channels (RGB)
#define KERNEL 3

// Number of filters
#define CONV1_FILTERS 16
#define CONV2_FILTERS 32

// Conv1 dimensions (with padding=1, size stays same before pooling)
#define CONV1_OUT_H 32
#define CONV1_OUT_W 32
#define POOL1_OUT_H 16
#define POOL1_OUT_W 16

// Conv2 dimensions (with padding=1, size stays same before pooling)
#define CONV2_OUT_H 16
#define CONV2_OUT_W 16
#define POOL2_OUT_H 8
#define POOL2_OUT_W 8

// Fully connected
#define FC_IN (CONV2_FILTERS * POOL2_OUT_H * POOL2_OUT_W)  // 32 * 8 * 8 = 2048
#define FC_OUT 10

/* ================== CNN LAYERS ================== */

// Convolution Layer 1 (3 input channels -> CONV1_FILTERS output channels)
// With padding=1
void conv2d_layer1(
    float input[IMG_C][IMG_H][IMG_W],
    float kernels[CONV1_FILTERS][IMG_C][KERNEL][KERNEL],
    float bias[CONV1_FILTERS],
    float output[CONV1_FILTERS][CONV1_OUT_H][CONV1_OUT_W]
) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < CONV1_OUT_H; i++) {
            for (int j = 0; j < CONV1_OUT_W; j++) {
                float sum = 0;
                // Sum over all input channels
                for (int c = 0; c < IMG_C; c++) {
                    for (int ki = 0; ki < KERNEL; ki++) {
                        for (int kj = 0; kj < KERNEL; kj++) {
                            // Handle padding
                            int in_i = i + ki - 1;
                            int in_j = j + kj - 1;
                            
                            if (in_i >= 0 && in_i < IMG_H && in_j >= 0 && in_j < IMG_W) {
                                sum += input[c][in_i][in_j] * kernels[f][c][ki][kj];
                            }
                        }
                    }
                }
                output[f][i][j] = sum + bias[f];
            }
        }
    }
}

// ReLU for Conv1 output
void relu1(
    float input[CONV1_FILTERS][CONV1_OUT_H][CONV1_OUT_W],
    float output[CONV1_FILTERS][CONV1_OUT_H][CONV1_OUT_W]
) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < CONV1_OUT_H; i++) {
            for (int j = 0; j < CONV1_OUT_W; j++) {
                output[f][i][j] = (input[f][i][j] > 0) ? input[f][i][j] : 0;
            }
        }
    }
}

// MaxPool 2x2 for Conv1 output
void maxpool1(
    float input[CONV1_FILTERS][CONV1_OUT_H][CONV1_OUT_W],
    float output[CONV1_FILTERS][POOL1_OUT_H][POOL1_OUT_W]
) {
    for (int f = 0; f < CONV1_FILTERS; f++) {
        for (int i = 0; i < POOL1_OUT_H; i++) {
            for (int j = 0; j < POOL1_OUT_W; j++) {
                float max_val = input[f][i*2][j*2];
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        float val = input[f][i*2 + pi][j*2 + pj];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                output[f][i][j] = max_val;
            }
        }
    }
}

// Convolution Layer 2 (CONV1_FILTERS input channels -> CONV2_FILTERS output channels)
// With padding=1
void conv2d_layer2(
    float input[CONV1_FILTERS][POOL1_OUT_H][POOL1_OUT_W],
    float kernels[CONV2_FILTERS][CONV1_FILTERS][KERNEL][KERNEL],
    float bias[CONV2_FILTERS],
    float output[CONV2_FILTERS][CONV2_OUT_H][CONV2_OUT_W]
) {
    for (int f_out = 0; f_out < CONV2_FILTERS; f_out++) {
        for (int i = 0; i < CONV2_OUT_H; i++) {
            for (int j = 0; j < CONV2_OUT_W; j++) {
                float sum = 0;
                // Sum over all input channels
                for (int f_in = 0; f_in < CONV1_FILTERS; f_in++) {
                    for (int ki = 0; ki < KERNEL; ki++) {
                        for (int kj = 0; kj < KERNEL; kj++) {
                            // Handle padding
                            int in_i = i + ki - 1;
                            int in_j = j + kj - 1;
                            
                            if (in_i >= 0 && in_i < POOL1_OUT_H && in_j >= 0 && in_j < POOL1_OUT_W) {
                                sum += input[f_in][in_i][in_j] * kernels[f_out][f_in][ki][kj];
                            }
                        }
                    }
                }
                output[f_out][i][j] = sum + bias[f_out];
            }
        }
    }
}

// ReLU for Conv2 output
void relu2(
    float input[CONV2_FILTERS][CONV2_OUT_H][CONV2_OUT_W],
    float output[CONV2_FILTERS][CONV2_OUT_H][CONV2_OUT_W]
) {
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < CONV2_OUT_H; i++) {
            for (int j = 0; j < CONV2_OUT_W; j++) {
                output[f][i][j] = (input[f][i][j] > 0) ? input[f][i][j] : 0;
            }
        }
    }
}

// MaxPool2 2x2 for Conv2 output
void maxpool2(
    float input[CONV2_FILTERS][CONV2_OUT_H][CONV2_OUT_W],
    float output[CONV2_FILTERS][POOL2_OUT_H][POOL2_OUT_W]
) {
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < POOL2_OUT_H; i++) {
            for (int j = 0; j < POOL2_OUT_W; j++) {
                float max_val = input[f][i*2][j*2];
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        float val = input[f][i*2 + pi][j*2 + pj];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                output[f][i][j] = max_val;
            }
        }
    }
}

// Flatten multi-channel feature maps
void flatten(
    float input[CONV2_FILTERS][POOL2_OUT_H][POOL2_OUT_W],
    float output[FC_IN]
) {
    int idx = 0;
    for (int f = 0; f < CONV2_FILTERS; f++) {
        for (int i = 0; i < POOL2_OUT_H; i++) {
            for (int j = 0; j < POOL2_OUT_W; j++) {
                output[idx++] = input[f][i][j];
            }
        }
    }
}

// Fully Connected Layer
void fully_connected(
    float input[FC_IN],
    float weights[FC_OUT][FC_IN],
    float bias[FC_OUT],
    float output[FC_OUT]
) {
    for (int i = 0; i < FC_OUT; i++) {
        float sum = 0;
        for (int j = 0; j < FC_IN; j++) {
            sum += input[j] * weights[i][j];
        }
        output[i] = sum + bias[i];
    }
}

// Argmax - find predicted class
int argmax(float input[FC_OUT]) {
    int idx = 0;
    float max_val = input[0];
    for (int i = 1; i < FC_OUT; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            idx = i;
        }
    }
    return idx;
}

/* ================== TOP CNN FUNCTION ================== */

int cnn_inference(
    float image[IMG_C][IMG_H][IMG_W],
    float conv1_kernels[CONV1_FILTERS][IMG_C][KERNEL][KERNEL],
    float conv1_bias[CONV1_FILTERS],
    float conv2_kernels[CONV2_FILTERS][CONV1_FILTERS][KERNEL][KERNEL],
    float conv2_bias[CONV2_FILTERS],
    float fc_weights[FC_OUT][FC_IN],
    float fc_bias[FC_OUT]
) {
    // Intermediate feature maps
    static float conv1_out[CONV1_FILTERS][CONV1_OUT_H][CONV1_OUT_W];
    static float pool1_out[CONV1_FILTERS][POOL1_OUT_H][POOL1_OUT_W];
    static float conv2_out[CONV2_FILTERS][CONV2_OUT_H][CONV2_OUT_W];
    static float pool2_out[CONV2_FILTERS][POOL2_OUT_H][POOL2_OUT_W];
    static float flat[FC_IN];
    static float fc_out[FC_OUT];

    // Forward pass
    conv2d_layer1(image, conv1_kernels, conv1_bias, conv1_out);
    relu1(conv1_out, conv1_out);
    maxpool1(conv1_out, pool1_out);

    conv2d_layer2(pool1_out, conv2_kernels, conv2_bias, conv2_out);
    relu2(conv2_out, conv2_out);
    maxpool2(conv2_out, pool2_out);

    flatten(pool2_out, flat);
    fully_connected(flat, fc_weights, fc_bias, fc_out);

    return argmax(fc_out);
}

/* ================== HELPER FUNCTIONS ================== */

// Helper function to load weights from .txt files
void load_weights(const string& filename, float* arr, int size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "ERROR: Could not open " << filename << "!" << endl;
        exit(1);
    }
    
    for (int i = 0; i < size; i++) {
        file >> arr[i];
    }
    
    file.close();
    cout << "Loaded: " << filename << endl;
}

/* ================== MAIN ================== */

int main() {
    // 1. Allocate arrays
    float image[IMG_C][IMG_H][IMG_W];
    
    float conv1_kernels[CONV1_FILTERS][IMG_C][KERNEL][KERNEL];
    float conv1_bias[CONV1_FILTERS];
    
    float conv2_kernels[CONV2_FILTERS][CONV1_FILTERS][KERNEL][KERNEL];
    float conv2_bias[CONV2_FILTERS];
    
    float fc_weights[FC_OUT][FC_IN];
    float fc_bias[FC_OUT];

    // 2. Load Model Weights
    cout << "=== Loading Model Weights === " << endl;
    load_weights("conv1_kernels.txt", (float*)conv1_kernels, CONV1_FILTERS * IMG_C * KERNEL * KERNEL);
    load_weights("conv1_bias.txt",    (float*)conv1_bias,    CONV1_FILTERS);
    load_weights("conv2_kernels.txt", (float*)conv2_kernels, CONV2_FILTERS * CONV1_FILTERS * KERNEL * KERNEL);
    load_weights("conv2_bias.txt",    (float*)conv2_bias,    CONV2_FILTERS);
    load_weights("fc_weights.txt",    (float*)fc_weights,    FC_OUT * FC_IN);
    load_weights("fc_bias.txt",       (float*)fc_bias,       FC_OUT);

    cout << "\n=== Measuring CPU Performance === " << endl;

    double total_time = 0;
    int correct_count = 0;

    // CIFAR-10 class names
    const string class_names[10] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    // Loop through 0-9
    for (int target_digit = 0; target_digit <= 9; target_digit++) {
        string filename = "test_image_label_" + to_string(target_digit) + ".txt";
        load_weights(filename, (float*)image, IMG_C * IMG_H * IMG_W);

        // --- START TIMER ---
        auto start = chrono::high_resolution_clock::now();    
        
        // Run Inference
        int prediction = cnn_inference(
            image, conv1_kernels, conv1_bias,
            conv2_kernels, conv2_bias,
            fc_weights, fc_bias
        );

        // --- STOP TIMER ---
        auto end = chrono::high_resolution_clock::now();
        
        // Calculate duration in milliseconds
        chrono::duration<double, milli> duration = end - start;
        total_time += duration.count();

        cout << "File: " << filename 
             << " | True: " << class_names[target_digit]
             << " | Pred: " << class_names[prediction]
             << " | Time: " << duration.count() << " ms";

        if (prediction == target_digit) {
            cout << " [SUCCESS]" << endl;
            correct_count++;
        } else {
            cout << " [FAIL]" << endl;
        }
    }

    cout << "\n----------------------------------------" << endl;
    cout << "   Accuracy: " << correct_count << "/10" << endl;
    cout << "   Avg Latency per Image: " << (total_time / 10.0) << " ms" << endl;
    cout << "----------------------------------------" << endl;
    
    return 0;
}