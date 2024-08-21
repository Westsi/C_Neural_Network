#include "activations.h"
#include <math.h>

float linear(float z) {
    return z;
}

float sigmoid(float z) {
    return 1 / (1 + expf(-z));
}

float logistic(float z) {
    return sigmoid(z);
}

float a_tanh(float z) {
    return tanhf(z);
}

float a_atan(float z) {
    return atanf(z);
}

float relu(float z) {
    if (z < 0) return 0;
    return z;
}

float leaky_relu(float z) {
    if (z < 0) return z * 0.01;
    return z;
}

// Modifies input in place; does not fit typedef
void softmax(float* z, int size) {
    float denom = 0;
    for (int i=0;i<size;i++) {
        denom = denom + expf(z[i]);
    }
    if (denom == 0) {
        denom = 0.0001;
    }
    for (int i=0;i<size;i++) {
        z[i] = expf(z[i]) / denom;
    }
}