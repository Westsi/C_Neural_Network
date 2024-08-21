#ifndef NEURON_H_
#define NEURON_H_
#include "activations.h"

typedef struct NEURON {
    activation_func_t activation;
    float value;
    int inputs;
    float* weights;
} neuron_t;

typedef neuron_t* neuron_ptr;

neuron_ptr newNeuron(activation_func_t act, int numInputs);
float calculateNeuronValue(neuron_ptr n, float* inputs);
void printNeuron(neuron_ptr n);

#endif