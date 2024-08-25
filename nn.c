#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "layer.h"
#include "activations.h"
#include "memtrack.h"
#include "network.h"

int main() {
    initNN();
    network_ptr network = newNetwork(newLayer(linear, -1, 10, INPUT_LAYER), 
                                     newLayer(softmax, 6, 6, OUTPUT_LAYER), 
                                     mse, 2, 
                                     newLayer(sigmoid, 10, 8, HIDDEN_LAYER), 
                                     newLayer(sigmoid, 8, 6, HIDDEN_LAYER));
    // network_ptr network = newNetwork(newLayer(linear, -1, 1, INPUT_LAYER), 
    //                                  newLayer(sigmoid, 1, 1, OUTPUT_LAYER), 
    //                                  mse, 2, 
    //                                  newLayer(sigmoid, 1, 1, HIDDEN_LAYER), 
    //                                  newLayer(sigmoid, 1, 1, HIDDEN_LAYER));
    printNetwork(network);
    float inp[10] = {0.7, 0.3, 0.8, 0.21, 0.1, 0.6, 0.3, 0.323, 0.0, 1.0};
    // float y[6] = {0.3971, 0.8, 0.1, 0.72, 0.8, 0.91};
    float y[6] = {0.2433, 0.12, 0.07, 0.158, 0.11, 0.2826};
    loadInputData(network->input, inp);
    for (int i=0;i<2500;i++) {
        printf("Epoch %d\n", i);
        float* result = forwardPass(network);
        for (int i=0;i<network->output->layerNeuronCnt;i++) {
            printf("Neuron %d: %.4f\n", i, result[i]);
        }
        printf("\n");
        printf("Cost: %.4f\n", network->cost(result, y, sizeof(y)/sizeof(y[0])));
        backprop(network, y);
    }
    printNetwork(network);
    for (int i=0;i<network->output->layerNeuronCnt;i++) {
            printf("Neuron %d: %.4f\n", i, network->output->neurons[i]->value);
        }
    freeAll();
}

/*
https://www.kaggle.com/code/hojjatk/read-mnist-dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
https://www.3blue1brown.com/lessons/backpropagation-calculus
https://docs.google.com/spreadsheets/d/18xI9C0xsbqvYZvH9DR7gSCWTYc6LZt_YQ_LdsjrDu0o/edit?gid=0#gid=0
*/