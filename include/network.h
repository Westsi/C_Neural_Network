#ifndef NETWORK_H_
#define NETWORK_H_
#include "layer.h"
#include "costs.h"

typedef struct NETWORK {
    layer_ptr input;
    layer_ptr* hidden;
    layer_ptr output;
    cost_func_t cost;
} network_t;

typedef network_t* network_ptr;

#endif