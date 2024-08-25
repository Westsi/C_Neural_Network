#include "layer.h"
#include "neuron.h"
#include "activations.h"
#include "memtrack.h"
#include <stdlib.h>
#include <stdio.h>

void* newLayer(activation_func_t act, int previousLayerNeurons, int layerNeurons, int layerType) {
    if (layerType == INPUT_LAYER) {
        input_layer_ptr dest = malloc(sizeof(input_layer_t));
        registerAllocated(dest);
        float* inpData = malloc(sizeof(float) * layerNeurons);
        registerAllocated(inpData);
        dest->layerCnt = layerNeurons;
        dest->data = inpData;
        return dest;
    }
    int actisSoft = 0;
    if (act == softmax) {
        actisSoft = 1;
    }
    layer_ptr dest = malloc(sizeof(layer_t));
    registerAllocated(dest);
    neuron_ptr* neurons = malloc(sizeof(neuron_ptr) * layerNeurons);
    registerAllocated(neurons);
    int cnt = 0;

    dest->activation = act;
    dest->actIsSoftmax = actisSoft;
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

void printInputLayer(input_layer_ptr layer) {
    printf("Input layer with %d values. Values: [\n", layer->layerCnt);
    for (int i=0;i<layer->layerCnt;i++) {
        if (i == layer->layerCnt - 1) {
            printf("\t%.2f\n", layer->data[i]);
            continue;
        }
        printf("%.2f, ", layer->data[i]);
    }
    printf("]\n");
}

void loadInputData(input_layer_ptr inp, float* data) {
    for (int i=0;i<inp->layerCnt;i++) {
        inp->data[i] = data[i];
    }
}

float* computeLayer(float* prevLayerData, layer_ptr layer) {
    float* layervals = malloc(sizeof(float) * layer->layerNeuronCnt);
    if (layer->layerType == OUTPUT_LAYER) {
        registerAllocated(layervals); // only if output because otherwise freed in other function
    }
    for (int i=0;i<layer->layerNeuronCnt;i++) {
        layervals[i] = calculateNeuronValue(layer->neurons[i], prevLayerData);
    }
    return layervals;
}

float* computeSoftmaxLayer(float* prevLayerData, layer_ptr layer) {
    float* layervals = malloc(sizeof(float) * layer->layerNeuronCnt);
    if (layer->layerType == OUTPUT_LAYER) {
        registerAllocated(layervals); // only if output because otherwise freed in other function
    }

    for (int i=0;i<layer->layerNeuronCnt;i++) {
        layervals[i] = calculateNeuronInput(layer->neurons[i], prevLayerData);
    }
    softmax(layervals, layer->layerNeuronCnt);

    for (int i=0;i<layer->layerNeuronCnt;i++) {
        layer->neurons[i]->value = layervals[i];
    }
    return layervals;
}