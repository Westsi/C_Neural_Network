#ifndef LAYER_H_
#define LAYER_H_

#include "neuron.h"
#include "activations.h"

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

typedef struct LAYER {
    activation_func_t activation;
    int actIsSoftmax;
    int layerNeuronCnt;
    int layerType;
    neuron_ptr* neurons;
} layer_t;

typedef layer_t* layer_ptr;

typedef struct INPUTLAYER {
    int layerCnt;
    float* data;
} input_layer_t;

typedef input_layer_t* input_layer_ptr;

void* newLayer(activation_func_t act, int previousLayerNeurons, int layerNeurons, int layerType);
void printLayer(layer_ptr layer);
void printInputLayer(input_layer_ptr layer);

void loadInputData(input_layer_ptr inp, float* data);
float* computeLayer(float* prevLayerData, layer_ptr layer);
float* computeSoftmaxLayer(float* prevLayerData, layer_ptr layer);

#endif