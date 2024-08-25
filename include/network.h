#ifndef NETWORK_H_
#define NETWORK_H_
#include "layer.h"
#include "costs.h"

typedef struct NETWORK {
    input_layer_ptr input;
    layer_ptr* hidden;
    layer_ptr output;
    int hiddenCnt;
    cost_func_t cost;
} network_t;

typedef network_t* network_ptr;

network_ptr newNetwork(input_layer_ptr input, layer_ptr output, cost_func_t cost, int hiddenCnt, ...);
void printNetwork(network_ptr network);
void initNN();
float* forwardPass(network_ptr net);
void backprop(network_ptr net, float* y);

#endif