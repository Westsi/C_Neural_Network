#include "neuron.h"
#include <stdlib.h>
#include "memtrack.h"
#include <stdio.h>

// the distance from 0 on either side that weights are initialised to (e.g. from -1 to 1 this would be 1)
#define RANDOMIZED_ZERO_DIST 1.0

// Randomize weights with values from -1 to 1 inclusive
void randomizeWeights(float* weights, int size) {
    for (int i=0;i<size;i++) {
        weights[i] = ((float)rand()/(float)RAND_MAX * (RANDOMIZED_ZERO_DIST*2)) - RANDOMIZED_ZERO_DIST;
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

    return dest;
}

float calculateNeuronValue(neuron_ptr n, float* inputs) {
    float v = 0;
    for (int i=0;i<n->inputs;i++) {
        v += inputs[i] * n->weights[i];
    }
    n->value = v;
    return v;
}

void printNeuron(neuron_ptr n) {
    printf("\tNeuron with %d inputs. Weights: [", n->inputs);
    for (int i=0;i<n->inputs;i++) {
        if (i == n->inputs-1) {
            printf("%.2f", n->weights[i]);
            continue;
        }
        printf("%.2f, ", n->weights[i]);
    }
    printf("]\n");
}