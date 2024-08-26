#include "neuron.h"
#include <stdlib.h>
#include "memtrack.h"
#include <stdio.h>

#define RANDOMIZED_ZERO_DIST 1.0

void randomizeWeights(float* weights, int size) {
    float std = sqrtf(2.0/(float)size);
    for (int i=0;i<size;i++) {
        weights[i] = (((float)rand()/(float)RAND_MAX * (RANDOMIZED_ZERO_DIST*2)) - RANDOMIZED_ZERO_DIST) * std;
    }
}


neuron_ptr newNeuron(activation_func_t act, int numInputs) {
    neuron_ptr dest = malloc(sizeof(neuron_t));
    registerAllocated(dest);
    float* weightsarr = malloc(sizeof(float) * numInputs);
    randomizeWeights(weightsarr, numInputs);
    registerAllocated(weightsarr);
    dest->activation = act;
    dest->value = 0;
    dest->weights = weightsarr;
    dest->inputs = numInputs;
    dest->bias = 0.1;

    return dest;
}

float calculateNeuronValue(neuron_ptr n, float* inputs) {
    float v = calculateNeuronInput(n, inputs);
    float a = n->activation(v);
    n->value = a;
    return a;
}

float calculateNeuronInput(neuron_ptr n, float* inputs) {
    float v = n->bias;
    for (int i=0;i<n->inputs;i++) {
        v += inputs[i] * n->weights[i];
    }
    n->calculatedInput = v;
    return v;
}

void printNeuron(neuron_ptr n) {
    printf("\tNeuron with %d ", n->inputs);
    if (n->inputs == 1) {
        printf("input. ");
    } else {
        printf("inputs. ");
    }
    printf("Bias: %.4f. Value: %.4f. Calculated Input: %.4f. Weights: [", n->bias, n->value, n->calculatedInput);
    for (int i=0;i<n->inputs;i++) {
        if (i == n->inputs-1) {
            printf("%.2f", n->weights[i]);
            continue;
        }
        printf("%.2f, ", n->weights[i]);
    }
    printf("]\n");
}