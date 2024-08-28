#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "network.h"
#include "memtrack.h"
#include "stdarg.h"
#include "defines.h"

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

    if (cost == categorical_cross_entropy && !dest->output->actIsSoftmax) {
        printf("ERROR!\nCategorical Cross Entropy Loss used but output does not use softmax activation!\n\n\n\n");
        exit(1);
    }
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

int isCorrect(float* pred, float* act, int size) {
    float maxPred = 0;
    int maxPredIdx = 11;
    int actIdx = 0;
    for (int i=0;i<size;i++) {
        if (pred[i] > maxPred) {
            maxPredIdx = i;
            maxPred = pred[i];
        }
        if (act[i] == 1) actIdx = i;
    }
    if (maxPredIdx == actIdx) return 1;
    return 0;
}

void train(network_ptr net, int epochs, data_callback_t trainData, data_callback_t testData, float lr) {
    int batchCnt = (int)TRAINING_SIZE/BATCH_SIZE + 1;
    // int batchCnt = 2;
    for (int i=0;i<epochs;i++) {
        printf("\nEpoch %d/%d\n", i+1, epochs);
        // batches
        printf("\tBatch 000/000");

        float batchCost = 0;
        int batchCorrect = 0;
        for (int b=0;b<batchCnt;b++) {

            /*
            float* result = forwardPass(net); 
            printf("Cost: %.4f\n", net->cost(result, y, sizeof(y)/sizeof(y[0])));
            backprop(net, y);
            */

            printf("\b\b\b\b\b\b\b"); // to remove the 000/000
            printf("%03d/%03d", b+1, batchCnt);
            fflush(stdout);
            batched_data_t batchedTrain = trainData(BATCH_SIZE);
            for (int d=0;d<batchedTrain.numBatched;d++) {
                loadInputData(net->input, batchedTrain.inputs[d]);
                float* result = forwardPass(net);
                batchCorrect += isCorrect(result, batchedTrain.outputs[d], NUM_OUTPUTS);
                float cost = net->cost(result, batchedTrain.outputs[d], NUM_OUTPUTS);
                // printf("\n\n\ncost %.4f\n", cost);
                batchCost += cost;
                // printf("Batch cost: %.4f\n", batchCost);
                backprop(net, batchedTrain.outputs[d], lr);
                // printf("BATCH LOSS: %.4f\n", batchCost);
                // printNetwork(net);

                // int ca = 0;
                // float maxPred = 0;
                // int mpi = 11;

                // for (int z=0;z<10;z++) {
                //     printf("%.2f \n", batchedTrain.outputs[d][z]);
                //     if (batchedTrain.outputs[d][z] == 1) {
                //         ca = z;
                //     }
                //     if (result[z] > maxPred) {
                //         maxPred = result[z];
                //         mpi = z;
                //     }
                // }

                // for (int c=0;c<28;c++) {
                //     for (int r=0;r<28;r++) {
                //         printf("%.3f ", (batchedTrain.inputs[d][(c*28)+r]));
                //     }
                //     printf("\n");
                // }
                // printf("Expected: %d, Got: %d, isCorrect: %d, cost: %.4f\n", ca, mpi, isCorrect(result, batchedTrain.outputs[d], NUM_OUTPUTS), cost);
                // if (d == 100) exit(1);
            }
        }
        // printf("BATCH LOSS: %.4f\n", batchCost);
        printf("\tLoss: %.4f - Accuracy: %.4f", (float)(batchCost/ (float)TRAINING_SIZE), (float)((float)batchCorrect / (float)TRAINING_SIZE));
        batched_data_t batchedTest = testData((int)(TEST_SIZE / epochs));
        float cost = 0;
        int correct = 0;
        for (int d=0;d<batchedTest.numBatched;d++) {
            loadInputData(net->input, batchedTest.inputs[d]);
            float* result = forwardPass(net);
            correct += isCorrect(result, batchedTest.outputs[d], NUM_OUTPUTS);
            cost += net->cost(result, batchedTest.outputs[d], NUM_OUTPUTS);
        }
        printf("\tValidation Loss: %.4f - Validation Accuracy: %.4f", (float)(cost / (float)batchedTest.numBatched), (float)((float)correct / (float)batchedTest.numBatched));
    }
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
    float* rval;
    if (net->output->actIsSoftmax) {
        rval = computeSoftmaxLayer(data, net->output);
    } else {
        rval = computeLayer(data, net->output);
    }
    free(data);
    return rval;
}

void backprop_v1(network_ptr net, float* y) {
    // go back one layer
    // derivative of cost function wrt activation of output neuron(s)
    // cost_deriv_t cder = getCostDerivative(net->cost);
    // derivative of activation function wrt to input to activation function (z)
    // activation_deriv_t ader = getActivationDerivative(net->output->activation);
    // derivative of z wrt to the weight, which comes out to the activation of the layer before
    // float zder = net->hidden[net->hiddenCnt-1]->neurons[0]->value;

    // float overallder = zder * ader(net->output->neurons[0]->calculatedInput) * cder(net->output->neurons[0]->value, y);
    // float biasder = ader(net->output->neurons[0]->calculatedInput) * cder(net->output->neurons[0]->value, y);

    // printf("∂C/∂w(L) = %.4f\n∂C/∂b(L) = %.4f\n", overallder, biasder);
    // apply change to weight
    // net->output->neurons[0]->weights[0] += -overallder;
    // net->output->neurons[0]->bias += -biasder;

    int nlayers = net->hiddenCnt + 1;
    layer_ptr layers[nlayers];
    int ind = 1;
    layers[0] = net->output;
    for (int i=net->hiddenCnt-1;i>=0;i--) {
        layers[ind++] = net->hidden[i];
    }
    // THIS IS WRONG! but idk what correct is
    cost_deriv_t cder = getCostDerivative(net->cost);
    float chainedDer = 1;
    for (int i=0;i<net->output->layerNeuronCnt;i++) {
        chainedDer += cder(net->output->neurons[i]->value, y[i], net->output->layerNeuronCnt);
    }

    for (int i=0;i<nlayers;i++) {
        activation_deriv_t ader = getActivationDerivative(layers[i]->activation);
        float adersum = 0;
        for (int j=0;j<layers[i]->layerNeuronCnt;j++) {
            float icd = chainedDer;
            float fader = ader(layers[i]->neurons[j]->calculatedInput);
            adersum += fader;
            icd = icd * fader;
            // at this point, icd is cder * ader so is ∂C wrt ∂b(L)_n
            // printf("∂C/∂b(%d)_%d = %.4f\n", nlayers-i, j, icd);
            layers[i]->neurons[j]->bias += -icd;
            for (int w=0;w<layers[i]->neurons[j]->inputs;w++) {
                float wicd = icd;
                if (i == nlayers-1) {
                    // idfk for now just breaking 
                    // this means these are the weights from input to first hidden
                    break;
                }
                // ∂C/∂w(L)_jk = a(L-1)_k
                wicd *= layers[i+1]->neurons[w]->value;
                // apply this to the weight
                // printf("∂C/∂w(%d)_%d<-%d = %.4f\n", nlayers-i, j, w, wicd);
                layers[i]->neurons[j]->weights[w] += -wicd;
            }

        }
        // is this correct method?
        chainedDer *= adersum;
    }
}

void backprop_v2(network_ptr net, float* y) {
    int nlayers = net->hiddenCnt + 1;
    layer_ptr layers[nlayers];
    int ind = 1;
    layers[0] = net->output;
    for (int i=net->hiddenCnt-1;i>=0;i--) {
        layers[ind++] = net->hidden[i];
    }

    cost_deriv_t cder = getCostDerivative(net->cost);
    float chainedDer[net->output->layerNeuronCnt];
    for (int i = 0; i < net->output->layerNeuronCnt; i++) {
        chainedDer[i] = cder(net->output->neurons[i]->value, y[i], net->output->layerNeuronCnt);
    }

    for (int i = 0; i < nlayers; i++) {
        activation_deriv_t ader = getActivationDerivative(layers[i]->activation);
        for (int j = 0; j < layers[i]->layerNeuronCnt; j++) {
            // Compute the error for this neuron based on its chainedDer
            float icd = chainedDer[j];
            printf("icd: %.4f\n", icd);
            float fader = ader(layers[i]->neurons[j]->calculatedInput);
            printf("fader: %.2f\n", fader);
            icd *= fader;
            printf("∂C/∂b(%d)_%d = %.4f\n", nlayers-i, j, icd);

            // Update the bias
            layers[i]->neurons[j]->bias += -icd;

            for (int w = 0; w < layers[i]->neurons[j]->inputs; w++) {
                float wicd = icd;
                if (i == nlayers - 1) {
                    wicd *= net->input->data[w];
                    printf("∂C/∂w(%d)_%d<-%d = %.4f\n", nlayers-i, j, w, wicd);
                    layers[i]->neurons[j]->weights[w] += -wicd;
                    continue;
                }
                // Backpropagate to update the weight
                wicd *= layers[i + 1]->neurons[w]->value;
                printf("∂C/∂w(%d)_%d<-%d = %.4f\n", nlayers-i, j, w, wicd);
                layers[i]->neurons[j]->weights[w] += -wicd;
            }
        }
        // Update chainedDer based on the sum of activations for the next layer
        for (int j = 0; j < layers[i]->layerNeuronCnt; j++) {
            printf("chained der before ader: %.2f\n", chainedDer[j]);
            chainedDer[j] *= ader(layers[i]->neurons[j]->calculatedInput);
            printf("chained der after ader: %.2f\n", chainedDer[j]);
        }
    }
}

void backprop(network_ptr net, float* y, float lr) {
    int nlayers = net->hiddenCnt + 1;
    layer_ptr layers[nlayers];
    int ind = 1;
    layers[0] = net->output;
    for (int i = net->hiddenCnt - 1; i >= 0; i--) {
        layers[ind++] = net->hidden[i];
    }


    cost_deriv_t cder = getCostDerivative(net->cost);
    float* chainedDer = (float*)malloc(net->output->layerNeuronCnt * sizeof(float));

    // Calculate delta for the output layer
    for (int i = 0; i < net->output->layerNeuronCnt; i++) {
        chainedDer[i] = cder(net->output->neurons[i]->value, y[i], net->output->layerNeuronCnt);
    }

    // Backpropagate through hidden layers
    for (int i = 0; i < nlayers; i++) {
        layer_ptr curr_layer = layers[i];
        activation_deriv_t ader = getActivationDerivative(curr_layer->activation);

        float* nextChainedDer = (i < nlayers - 1) ? (float*)malloc(layers[i + 1]->layerNeuronCnt * sizeof(float)) : NULL;

        // Compute deltas for the current layer
        // TODO: gpu multithread this loop
        for (int j = 0; j < curr_layer->layerNeuronCnt; j++) {
            float icd = chainedDer[j];
            float fader = ader(curr_layer->neurons[j]->calculatedInput);
            icd *= fader;

            // Update the bias
            curr_layer->neurons[j]->bias -= icd * lr;
            for (int w = 0; w < curr_layer->neurons[j]->inputs; w++) {
                float wicd = icd;
                if (i == nlayers - 1) {
                    // Update weights in the final layer
                    wicd *= net->input->data[w];
                } else {
                    // Update weights in hidden layers
                    wicd *= layers[i + 1]->neurons[w]->value;
                }
                curr_layer->neurons[j]->weights[w] -= wicd * lr;
            }
        }

        // Update chainedDer for the next layer
        if (nextChainedDer != NULL) {
            for (int j = 0; j < layers[i + 1]->layerNeuronCnt; j++) {
                float delta_sum = 0;
                for (int k = 0; k < curr_layer->layerNeuronCnt; k++) {
                    if (k < layers[i + 1]->neurons[j]->inputs) {
                        delta_sum += chainedDer[k] * layers[i + 1]->neurons[k]->weights[j];
                    }
                }
                nextChainedDer[j] = delta_sum * ader(layers[i + 1]->neurons[j]->calculatedInput);
            }
            free(chainedDer);
            chainedDer = nextChainedDer;
        }
    }

    free(chainedDer);
}


float** oneHotEncode(uint8_t* vals, int size, int nOutputs) {
    float** oneHotEncoded = malloc(sizeof(float*) * size);
    registerAllocated(oneHotEncoded);
    for (int i=0;i<size;i++) {
        float* outputs = calloc(sizeof(float), nOutputs);
        registerAllocated(outputs);
        int v = vals[i];
        // printf("%d: %d\n", i, v);
        outputs[v] = 1;
        oneHotEncoded[i] = outputs;
    }
    return oneHotEncoded;
}
