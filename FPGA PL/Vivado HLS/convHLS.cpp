/*
 * Conv2D - Fixed-Point Optimized (Zynq-7020 Friendly)
 * - Internal compute uses fixed-point
 * - AXI interfaces remain float
 * - II = 4
 * - Kernel partial unroll = 3
 */

#include <ap_fixed.h>

#define MAX_PIXELS  32768
#define MAX_WEIGHTS 4608
#define MAX_COUT    32

// ================= Fixed-point types =================
typedef ap_fixed<16,6, AP_RND, AP_SAT> data_t;
typedef ap_fixed<32,12, AP_RND, AP_SAT> acc_t;

extern "C" {

void convHLS(
    const float *input,
    const float *weights,
    const float *bias,
    float *output,
    int Cin,
    int Cout,
    int H,
    int W
) {
    // =========================================================
    // INTERFACE
    // =========================================================
    #pragma HLS INTERFACE m_axi port=input   offset=slave bundle=gmem0 depth=16384
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem0 depth=4608
    #pragma HLS INTERFACE m_axi port=bias    offset=slave bundle=gmem0 depth=32
    #pragma HLS INTERFACE m_axi port=output  offset=slave bundle=gmem1 depth=32768

    #pragma HLS INTERFACE s_axilite port=Cin    bundle=control
    #pragma HLS INTERFACE s_axilite port=Cout   bundle=control
    #pragma HLS INTERFACE s_axilite port=H      bundle=control
    #pragma HLS INTERFACE s_axilite port=W      bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // =========================================================
    // LOCAL BUFFERS (Fixed-point)
    // =========================================================
    static data_t in_buf[16384];
    #pragma HLS RESOURCE variable=in_buf core=RAM_2P_BRAM

    static data_t w_buf[4608];
    #pragma HLS RESOURCE variable=w_buf core=RAM_2P_BRAM

    static data_t out_buf[32768];
    #pragma HLS RESOURCE variable=out_buf core=RAM_2P_BRAM

    static acc_t bias_local[MAX_COUT];
    #pragma HLS ARRAY_PARTITION variable=bias_local complete

    // =========================================================
    // CONSTANTS
    // =========================================================
    const int HW    = H * W;
    const int KK    = 9;
    const int CinKK = Cin * KK;

    // =========================================================
    // LOAD INPUT (float → fixed)
    // =========================================================
    Load_Input:
    for (int i = 0; i < Cin * HW; i++) {
        #pragma HLS PIPELINE II=1
        in_buf[i] = (data_t)input[i];
    }

    // =========================================================
    // LOAD WEIGHTS (float → fixed)
    // =========================================================
    Load_Weights:
    for (int i = 0; i < Cout * CinKK; i++) {
        #pragma HLS PIPELINE II=1
        w_buf[i] = (data_t)weights[i];
    }

    // =========================================================
    // LOAD BIAS (float → fixed)
    // =========================================================
    Load_Bias:
    for (int i = 0; i < Cout; i++) {
        #pragma HLS PIPELINE II=1
        bias_local[i] = (acc_t)bias[i];
    }

    // =========================================================
    // COMPUTE
    // =========================================================
    Compute_OC:
    for (int oc = 0; oc < Cout; oc++) {

        const int w_base   = oc * CinKK;
        const int out_base = oc * HW;

        Compute_Spatial:
        for (int hw = 0; hw < HW; hw++) {
            #pragma HLS PIPELINE II=4

            const int i = hw / W;
            const int j = hw % W;

            acc_t sum = bias_local[oc];

            Compute_IC:
            for (int ic = 0; ic < Cin; ic++) {

                const int in_base   = ic * HW;
                const int w_ic_base = w_base + ic * KK;

                int kidx = 0;

                Compute_KH:
                for (int ki = 0; ki < 3; ki++) {
                    const int ii = i + ki - 1;

                    Compute_KW:
                    for (int kj = 0; kj < 3; kj++, kidx++) {
                        #pragma HLS UNROLL factor=3

                        const int jj = j + kj - 1;

                        data_t pixel = 0;
                        if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                            pixel = in_buf[in_base + ii * W + jj];
                        }

                        sum += (acc_t)pixel * (acc_t)w_buf[w_ic_base + kidx];
                    }
                }
            }

            // ReLU
            out_buf[out_base + hw] = (sum > 0) ? (data_t)sum : (data_t)0;
        }
    }

    // =========================================================
    // WRITE OUTPUT (fixed → float)
    // =========================================================
    Write_Output:
    for (int i = 0; i < Cout * HW; i++) {
        #pragma HLS PIPELINE II=1
        output[i] = (float)out_buf[i];
    }
}

} // extern "C"
