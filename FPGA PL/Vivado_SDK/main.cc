/*
 * CIFAR-10 Hybrid Accelerator - Updated for Fixed-Point HLS IP
 * FPGA: Conv1, Conv2 (Fixed-point internally, float AXI)
 * CPU:  Pool1, Pool2, FC (Dense)
 * Data: Uses wfc_raw / bfc_raw from data.h
 */

#include <stdio.h>
#include <stdlib.h>
#include "xparameters.h"
#include "xconvhls.h"        // Updated to match new IP name
#include "xtime_l.h"
#include "xil_cache.h"
#include "data.h"

#define IMG_H 32
#define IMG_W 32
#define IMG_C 3
#define CONV1_OUT_CH 16
#define CONV2_OUT_CH 32
#define K 3
#define FC_IN 2048   // 32 channels * 8 * 8
#define FC_OUT 10    // 10 Classes

// =========================================================
// BUFFERS (Aligned for Cache Safety)
// =========================================================
static float conv1_out[CONV1_OUT_CH * IMG_H * IMG_W] __attribute__ ((aligned (64)));
static float pool1_out[CONV1_OUT_CH * 16 * 16]       __attribute__ ((aligned (64)));
static float conv2_out[CONV2_OUT_CH * 16 * 16]       __attribute__ ((aligned (64)));
static float pool2_out[CONV2_OUT_CH * 8 * 8]         __attribute__ ((aligned (64)));
static float fc_out[FC_OUT];

// =========================================================
// CPU LAYERS
// =========================================================

// MaxPool 2x2
void maxpool_2x2(int channels, int in_h, int in_w, float *input, float *output) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max_val = -1e9;
                for (int pi = 0; pi < 2; pi++) {
                    for (int pj = 0; pj < 2; pj++) {
                        int in_idx = c * (in_h * in_w) + (i*2 + pi) * in_w + (j*2 + pj);
                        if (input[in_idx] > max_val) max_val = input[in_idx];
                    }
                }
                int out_idx = c * (out_h * out_w) + i * out_w + j;
                output[out_idx] = max_val;
            }
        }
    }
}

// Fully Connected Layer (Using wfc_raw / bfc_raw)
void fc_layer_cpu(float *input, float *output) {
    for (int i = 0; i < FC_OUT; i++) {
        float sum = bfc_raw[i]; // Load Bias
        for (int j = 0; j < FC_IN; j++) {
            // Weights are stored as [Output][Input] flat array
            sum += input[j] * wfc_raw[i * FC_IN + j];
        }
        output[i] = sum;
    }
}

// Argmax to find predicted class
int argmax(float *scores, int n) {
    int max_idx = 0;
    float max_val = scores[0];
    for (int i = 1; i < n; i++) {
        if (scores[i] > max_val) {
            max_val = scores[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// =========================================================
// MAIN
// =========================================================
int main() {
    printf("\n=== CIFAR-10 FPGA Accelerator (Fixed-Point) ===\n");

    // 1. Initialize Hardware
    XConvhls conv_ip;  // Updated struct name
    XConvhls_Config *cfg = XConvhls_LookupConfig(XPAR_CONVHLS_0_DEVICE_ID);
    if (!cfg) {
        printf("Error: Config not found\n");
        return -1;
    }
    if (XConvhls_CfgInitialize(&conv_ip, cfg) != XST_SUCCESS) {
        printf("Error: Init failed\n");
        return -1;
    }

    printf("Hardware initialized (Fixed-Point IP). Starting Inference...\n\n");

    Xil_DCacheFlush();

    const char *class_names[10] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    // Standard CIFAR-10 test set labels for the first 10 images
    int labels[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int correct_count = 0;
    double total_latency = 0;
    XTime tStart, tEnd;

    // 2. Inference Loop
    for (int img = 0; img < 10; img++) {
        float *input_img;

        // Select Image from data.h
        switch(img) {
            case 0: input_img = (float*)img0_raw; break;
            case 1: input_img = (float*)img1_raw; break;
            case 2: input_img = (float*)img2_raw; break;
            case 3: input_img = (float*)img3_raw; break;
            case 4: input_img = (float*)img4_raw; break;
            case 5: input_img = (float*)img5_raw; break;
            case 6: input_img = (float*)img6_raw; break;
            case 7: input_img = (float*)img7_raw; break;
            case 8: input_img = (float*)img8_raw; break;
            case 9: input_img = (float*)img9_raw; break;
            default: input_img = (float*)img0_raw; break;
        }

        // Flush Input & Conv Weights to DRAM
        Xil_DCacheFlushRange((INTPTR)input_img, IMG_C * IMG_H * IMG_W * sizeof(float));
        Xil_DCacheFlushRange((INTPTR)w1_raw, CONV1_OUT_CH * IMG_C * K * K * sizeof(float));
        Xil_DCacheFlushRange((INTPTR)b1_raw, CONV1_OUT_CH * sizeof(float));
        Xil_DCacheFlushRange((INTPTR)w2_raw, CONV2_OUT_CH * CONV1_OUT_CH * K * K * sizeof(float));
        Xil_DCacheFlushRange((INTPTR)b2_raw, CONV2_OUT_CH * sizeof(float));

        XTime_GetTime(&tStart);

        // --- LAYER 1: FPGA (Conv) ---
        // Updated API function names to match new IP
        XConvhls_Set_input_r(&conv_ip, (u64)input_img);
        XConvhls_Set_weights(&conv_ip, (u64)w1_raw);
        XConvhls_Set_bias(&conv_ip, (u64)b1_raw);
        XConvhls_Set_output_r(&conv_ip, (u64)conv1_out);
        XConvhls_Set_Cin(&conv_ip, IMG_C);
        XConvhls_Set_Cout(&conv_ip, CONV1_OUT_CH);
        XConvhls_Set_H(&conv_ip, IMG_H);
        XConvhls_Set_W(&conv_ip, IMG_W);
        XConvhls_Start(&conv_ip);
        while (!XConvhls_IsDone(&conv_ip));

        // Invalidate cache to read results from DDR
        Xil_DCacheInvalidateRange((INTPTR)conv1_out, CONV1_OUT_CH * IMG_H * IMG_W * sizeof(float));

        // --- LAYER 1: CPU (Pool) ---
        maxpool_2x2(CONV1_OUT_CH, 32, 32, conv1_out, pool1_out);
        Xil_DCacheFlushRange((INTPTR)pool1_out, CONV1_OUT_CH * 16 * 16 * sizeof(float));

        // --- LAYER 2: FPGA (Conv) ---
        XConvhls_Set_input_r(&conv_ip, (u64)pool1_out);
        XConvhls_Set_weights(&conv_ip, (u64)w2_raw);
        XConvhls_Set_bias(&conv_ip, (u64)b2_raw);
        XConvhls_Set_output_r(&conv_ip, (u64)conv2_out);
        XConvhls_Set_Cin(&conv_ip, CONV1_OUT_CH);
        XConvhls_Set_Cout(&conv_ip, CONV2_OUT_CH);
        XConvhls_Set_H(&conv_ip, 16);
        XConvhls_Set_W(&conv_ip, 16);
        XConvhls_Start(&conv_ip);
        while (!XConvhls_IsDone(&conv_ip));

        Xil_DCacheInvalidateRange((INTPTR)conv2_out, CONV2_OUT_CH * 16 * 16 * sizeof(float));

        // --- LAYER 2: CPU (Pool) ---
        maxpool_2x2(CONV2_OUT_CH, 16, 16, conv2_out, pool2_out);

        // --- LAYER 3: CPU (Fully Connected) ---
        fc_layer_cpu(pool2_out, fc_out);

        XTime_GetTime(&tEnd);

        // --- Stats & Accuracy ---
        double ms = 1000.0 * (double)(tEnd - tStart) / (double)COUNTS_PER_SECOND;
        total_latency += ms;

        int prediction = argmax(fc_out, 10);
        int truth = labels[img];

        printf("Image %d: Actual = %-10s | Pred = %-10s | Time = %.2f ms ",
               img, class_names[truth], class_names[prediction], ms);

        if (prediction == truth) {
            printf("[PASS]\n");
            correct_count++;
        } else {
            printf("[FAIL]\n");
        }
    }

    printf("\n============================================\n");
    printf("FINAL ACCURACY: %d / 10 (%.1f%%)\n", correct_count, (float)correct_count * 10.0);
    printf("AVG LATENCY:    %.3f ms\n", total_latency / 10.0);
    printf("============================================\n");

    return 0;
}
