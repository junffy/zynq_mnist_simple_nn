#include <stdio.h>
#include <ap_fixed.h>

#include "af1_weight.h"
#include "af1_bias.h"
#include "af2_weight.h"
#include "af2_bias.h"
#include "mnist_data.h"

int mnist_nn(ap_ufixed<8, 0, AP_TRN_ZERO, AP_SAT> in[784], ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> out[10]);
int mnist_nn_float(float in[784], float out[10]);
int max_ap_fixed(ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> out[10]);
int max_float(float out[10]);

#define NUM_ITERATIONS    100 // C Simulation
// #define NUM_ITERATIONS    2 // C/RTL CoSimulation

int main(){
    float t_tran_float[NUM_ITERATIONS][784];
    ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> result_ap_fixed[NUM_ITERATIONS][10];
    float result_float[NUM_ITERATIONS][10];
    int max_id_hw, max_id_sw, max_id_ref;

    for(int i=0; i<NUM_ITERATIONS; i++)
        for(int j=0; j<784; j++)
            t_tran_float[i][j] = (float)t_train[i][j];

    for(int i=0; i<NUM_ITERATIONS; i++){
        mnist_nn(&t_train[i][0], &result_ap_fixed[i][0]);
        mnist_nn_float(&t_tran_float[i][0], &result_float[i][0]);
    }

    int errflag=0;
    for(int i=0; i<NUM_ITERATIONS; i++){
        max_id_hw = max_ap_fixed(&result_ap_fixed[i][0]);
        max_id_sw = max_float(&result_float[i][0]);
        max_id_ref = max_float(&t_test[i][0]);

        if(max_id_ref != max_id_hw){
            printf("id = %d, max_id_ref = %d, max_id_hw = %d\n", i, max_id_ref, max_id_hw);
            errflag = 1;
        }
        if(max_id_ref != max_id_sw){
            printf("id = %d, max_id_ref = %d, max_id_sw = %d\n", i, max_id_ref, max_id_sw);
            errflag = 1;
        }
    }
    if(errflag == 0)
        printf("No Error\n");

    return(0);
}

int mnist_nn_float(float in[784], float out[10]){
    float dot1[50];
    float dot2[10];

    af1_dot1: for(int col=0; col<50; col++){
        dot1[col] = 0;
        af1_dot2: for(int row=0; row<784; row++){
            dot1[col] += in[row]*af1_fweight[row][col];
        }
        dot1[col] += af1_fbias[col];

        if(dot1[col] < 0)    // ReLU
            dot1[col] = 0;
    }

    af2_dot1: for(int col=0; col<10; col++){
        dot2[col] = 0;
        af2_dot2: for(int row=0; row<50; row++){
            dot2[col] += dot1[row]*af2_fweight[row][col];
        }
        dot2[col] += af2_fbias[col];

        if(dot2[col] < 0)    // ReLU
            dot2[col] = 0;
        out[col] = dot2[col];
    }

    return(0);
}

int max_ap_fixed(ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> out[10]){
    int max_id;
    ap_fixed<13, 5, AP_TRN_ZERO, AP_SAT> max;

    for(int i=0; i<10; i++){
        if(i == 0){
            max = out[0];
            max_id = 0;
        }else if(out[i]>max){
            max = out[i];
            max_id = i;
        }
    }
    return(max_id);
}

int max_float(float out[10]){
    int max_id;
    float max;

    for(int i=0; i<10; i++){
        if(i == 0){
            max = out[0];
            max_id = 0;
        }else if(out[i]>max){
            max = out[i];
            max_id = i;
        }
    }
    return(max_id);
}

