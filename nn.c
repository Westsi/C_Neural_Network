#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "layer.h"
#include "activations.h"
#include "memtrack.h"
#include "network.h"

int main() {
    initNN();
    // network_ptr network = newNetwork(newLayer(linear, -1, 2, INPUT_LAYER), 
    //                                  newLayer(sigmoid, 1, 1, OUTPUT_LAYER), 
    //                                  mse, 2, 
    //                                  newLayer(sigmoid, 2, 2, HIDDEN_LAYER), 
    //                                  newLayer(sigmoid, 2, 1, HIDDEN_LAYER));
    network_ptr network = newNetwork(newLayer(linear, -1, 1, INPUT_LAYER), 
                                     newLayer(sigmoid, 1, 1, OUTPUT_LAYER), 
                                     mse, 2, 
                                     newLayer(sigmoid, 1, 1, HIDDEN_LAYER), 
                                     newLayer(sigmoid, 1, 1, HIDDEN_LAYER));
    printNetwork(network);
    // float inp[2] = {0.7, 0.3};
    float inp[1] = {0.7};
    float y[1] = {0.6};
    loadInputData(network->input, inp);
    for (int i=0;i<500;i++) {
        float* result = forwardPass(network);
        for (int i=0;i<network->output->layerNeuronCnt;i++) {
            printf("Neuron %d: %.3f", i, result[i]);
        }
        printf("\n");
        printf("Cost: %.3f\n", network->cost(result, y, 1));
        backprop(network, y[0]);
    }
    printNetwork(network);

    freeAll();
}

/*
https://www.kaggle.com/code/hojjatk/read-mnist-dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
https://www.bmc.com/blogs/neural-network-introduction/
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
https://www.3blue1brown.com/lessons/backpropagation-calculus
https://docs.google.com/spreadsheets/d/18xI9C0xsbqvYZvH9DR7gSCWTYc6LZt_YQ_LdsjrDu0o/edit?gid=0#gid=0
https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
https://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables
*/