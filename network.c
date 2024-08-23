#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "network.h"
#include "memtrack.h"
#include "stdarg.h"

network_ptr newNetwork(input_layer_ptr input, layer_ptr output, cost_func_t cost, int hiddenCnt, ...) {
    va_list ptr;
    va_start(ptr, hiddenCnt);

    network_ptr dest = malloc(sizeof(network_t));
    registerAllocated(dest);
    layer_ptr* hiddenarray = malloc(sizeof(layer_ptr) * hiddenCnt);
    registerAllocated(hiddenarray);
    dest->hidden = hiddenarray;
    dest->input = input;
    dest->output = output;
    dest->cost = cost;
    dest->hiddenCnt = hiddenCnt;

    for (int i=0;i<hiddenCnt;i++) {
        hiddenarray[i] = va_arg(ptr, layer_ptr);
    }
    va_end(ptr);
    return dest;
}

void printNetwork(network_ptr network) {
    printf("--Network--\n");
    printInputLayer(network->input);
    printf("%d Hidden Layers:\n", network->hiddenCnt);
    for (int i=0;i<network->hiddenCnt;i++) {
        printLayer(network->hidden[i]);
    }
    printf("Output Layer:\n");
    printLayer(network->output);
    printf("Cost Function: %p\n", network->cost);
}

void initNN() {
    srand((unsigned int)time(NULL));
}

float* forwardPass(network_ptr net) {
    if (net->hiddenCnt < 1) {
        return computeLayer(net->input->data, net->output);
    }

    float* data = net->input->data;
    if (net->hiddenCnt > 0) {
        for (int i=0;i<net->hiddenCnt;i++) {
            float* retdata = computeLayer(data, net->hidden[i]);
            if (i != 0) {
                free(data);
            }
            data = retdata;
        }
    }
    return computeLayer(data, net->output);
}

void backprop(network_ptr net, float y) {
    // go back one layer
    // derivative of cost function wrt activation of output neuron(s)
    cost_deriv_t cder = getCostDerivative(net->cost);
    // derivative of activation function wrt to input to activation function (z)
    activation_deriv_t ader = getActivationDerivative(net->output->activation);
    // derivative of z wrt to the weight, which comes out to the activation of the layer before
    float zder = net->hidden[net->hiddenCnt-1]->neurons[0]->value;

    float overallder = zder * ader(net->output->neurons[0]->calculatedInput) * cder(net->output->neurons[0]->value, y);
    printf("∂C/∂w(L) = %.4f\n", overallder);
    // apply change to weight
    net->output->neurons[0]->weights[0] += -overallder;
}
