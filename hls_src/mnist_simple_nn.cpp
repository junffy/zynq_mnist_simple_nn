#include <stdio.h>
#include <ap_fixed.h>

#include "af1_weight.h"
#include "af1_bias.h"
#include "af2_weight.h"
#include "af2_bias.h"


int mnist_nn(ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> in[784], ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> out[10]){
#pragma HLS INTERFACE s_axilite port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=return
    ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> buf[784];
    ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> dot1[50];
    ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> dot2[10];

    buf_copy: for(int i=0; i<784; i++)
        buf[i] = in[i];

    af1_dot1: for(int col=0; col<50; col++){
        dot1[col] = 0;
        af1_dot2: for(int row=0; row<784; row++){
            dot1[col] += buf[row]*af1_weight[row][col];
        }
        dot1[col] += af1_bias[col];

        if(dot1[col] < 0)    // ReLU
            dot1[col] = 0;
    }

    af2_dot1: for(int col=0; col<10; col++){
        dot2[col] = 0;
        af2_dot2: for(int row=0; row<50; row++){
            dot2[col] += dot1[row]*af2_weight[row][col];
        }
        dot2[col] += af2_bias[col];

        if(dot2[col] < 0)    // ReLU
            dot2[col] = 0;
        out[col] = dot2[col];
    }

    return(0);
}

