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