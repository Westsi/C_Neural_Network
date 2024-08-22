#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "layer.h"
#include "activations.h"
#include "memtrack.h"

void init_nn() {
    srand((unsigned int)time(NULL));
}

int main() {
    init_nn();
    layer_ptr layer = newLayer(sigmoid, 4, 4, HIDDEN_LAYER);
    printLayer(layer);
    freeAll();
}

/*
https://www.kaggle.com/code/hojjatk/read-mnist-dataset
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
https://www.bmc.com/blogs/neural-network-introduction/
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
https://www.3blue1brown.com/lessons/backpropagation-calculus
*/