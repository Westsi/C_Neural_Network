#include "layer.h"
#include "neuron.h"
#include "activations.h"
#include "memtrack.h"
#include <stdlib.h>
#include <stdio.h>

layer_ptr newLayer(activation_func_t act, int previousLayerNeurons, int layerNeurons, int layerType) {
    layer_ptr dest = malloc(sizeof(layer_t));
    registerAllocated(dest);
    neuron_ptr* neurons = malloc(sizeof(neuron_ptr) * layerNeurons);
    registerAllocated(neurons);
    int cnt = 0;

    dest->activation = act;
    dest->layerNeuronCnt = layerNeurons;
    dest->layerType = layerType;

    for (int i=0;i<layerNeurons;i++) {
        neurons[cnt++] = newNeuron(act, previousLayerNeurons);
    }

    dest->neurons = neurons;

    return dest;
}

void printLayer(layer_ptr layer) {
    switch (layer->layerType) {
        case INPUT_LAYER:
            printf("Input layer ");
            break;
        case HIDDEN_LAYER:
            printf("Hidden layer ");
            break;
        case OUTPUT_LAYER:
            printf("Output layer ");
            break;
    }

    printf("with %d neurons. Activation: %p, Neurons: [\n", layer->layerNeuronCnt, layer->activation);

    for (int i=0;i<layer->layerNeuronCnt;i++) {
        printNeuron(layer->neurons[i]);
    }

    printf("]\n");
}