#ifndef NETWORK_H_
#define NETWORK_H_
#include "layer.h"
#include "costs.h"
#include <stdint.h>


typedef struct NETWORK {
    input_layer_ptr input;
    layer_ptr* hidden;
    layer_ptr output;
    int hiddenCnt;
    cost_func_t cost;
} network_t;

typedef struct BATCHEDDATA {
    float** inputs;
    int numBatched;
    float** outputs; 
} batched_data_t;

typedef network_t* network_ptr;

typedef batched_data_t (*data_callback_t) (int);

network_ptr newNetwork(input_layer_ptr input, layer_ptr output, cost_func_t cost, int hiddenCnt, ...);
void printNetwork(network_ptr network);
void saveNetwork(network_ptr network, int epoch);
void train(network_ptr net, int epochs, data_callback_t trainData, data_callback_t testData, float lr);
void initNN();
float* forwardPass(network_ptr net);
void backprop(network_ptr net, float* y, float lr);

float** oneHotEncode(uint8_t* vals, int size, int nOutputs);

#endif