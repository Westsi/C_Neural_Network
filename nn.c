#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "layer.h"
#include "activations.h"
#include "memtrack.h"
#include "network.h"
#include "mnistreader.h"

float** trainingData;
float** oneHotTrainingLabels;
float** testData;
float** oneHotTestLabels;

batched_data_t mnistTrainingBatcher(int nextN) {
    static int index = 0;
    batched_data_t r;
    r.inputs = trainingData + (index);
    r.outputs = oneHotTrainingLabels + (index);
    r.numBatched = nextN;
    if (index + nextN > 60000) {
        r.numBatched = 60000 - index;
        index = 0;
    } else {
        index += nextN;
    }
    return r;
}

batched_data_t mnistTestBatcher(int nextN) {
    static int index = 0;
    batched_data_t r;
    r.inputs = testData + (index);
    r.outputs = oneHotTestLabels + (index);
    r.numBatched = nextN;
    if (index + nextN > 10000) {
        r.numBatched = 10000 - index;
        index = 0;
    } else {
        index += nextN;
    }
    return r;
}

void valgrindCheck() {
    network_ptr network = newNetwork(newLayer(linear, -1, 10, INPUT_LAYER), 
                                     newLayer(softmax, 4, 2, OUTPUT_LAYER), 
                                     mse, 2, 
                                     newLayer(sigmoid, 10, 8, HIDDEN_LAYER),
                                     newLayer(sigmoid, 8, 4, HIDDEN_LAYER));
    float x[10] = {0.5, 0.6, 0.3, 0.89, 0.1, 0.0, 0.9, 0.56, 0.39, 0.2};
    float y[2] = {0.32, 0.68};
    for (int i=0;i<10;i++) {
        loadInputData(network->input, x);
        float* result = forwardPass(network);
        float cost = network->cost(result, y, 2);
        printf("1: %.4f, 2: %.4f\n", result[0], result[1]);
        printf("Cost of Epoch %d: %.4f\n", i, cost);
        backprop(network, y, 1);
    }
    printNetwork(network);
    freeAll();
}

int main() {
    #ifdef VALGRIND
        valgrindCheck();
        return;
    #endif
    initMnist();
    trainingData = readTrainingData();
    testData = readTestData();
    oneHotTrainingLabels = oneHotEncode(readTrainingLabels(), 60000, 10);
    oneHotTestLabels = oneHotEncode(readTestLabels(), 10000, 10);
    initNN();
    network_ptr network = newNetwork(newLayer(linear, -1, 784, INPUT_LAYER), 
                                     newLayer(softmax, 128, 10, OUTPUT_LAYER), 
                                     categorical_cross_entropy, 1, 
                                     newLayer(relu, 784, 128, HIDDEN_LAYER));
    // printNetwork(network);
    train(network, 6, mnistTrainingBatcher, mnistTestBatcher, 0.01);
    freeAll();
    closeAll();
}

/*
https://www.kaggle.com/code/hojjatk/read-mnist-dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
https://www.3blue1brown.com/lessons/backpropagation-calculus
https://docs.google.com/spreadsheets/d/18xI9C0xsbqvYZvH9DR7gSCWTYc6LZt_YQ_LdsjrDu0o/edit?gid=0#gid=0


https://stackoverflow.com/questions/17598572/how-to-read-write-a-binary-file
https://www.tensorflow.org/datasets/keras_example
https://www.w3schools.com/c/c_files_read.php
*/