#include "activations.h"
#include <math.h>

inline float linear(float z) {
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

inline float deriv_linear(float a) {
    return 1;
}

activation_deriv_t getActivationDerivative(activation_func_t a) {
    if (a == sigmoid) {
        return deriv_sigmoid;
    }
    if (a == logistic) {
        return deriv_logistic;
    }
    if (a == a_tanh) {
        return deriv_a_tanh;
    }
    if (a == a_atan) {
        return deriv_a_atan;
    }
    if (a == relu) {
        return deriv_relu;
    }
    if (a == leaky_relu) {
        return deriv_leaky_relu;
    }
    return deriv_linear;
}

float deriv_sigmoid(float a) {
    float sig_a = sigmoid(a);
    float res = sig_a * (1 - sig_a);
    return res;
}

float deriv_logistic(float a) {
    return deriv_sigmoid(a);
}

float deriv_a_tanh(float a) {
    float tanh_a = a_tanh(a);
    float res = 1.0 - (tanh_a*tanh_a);
    return res;
}

float deriv_a_atan(float a) {
    float res = 1 / ((a * a) + 1);
    return res;
}

float deriv_relu(float a) {
    if (a < 0) return 0;
    return 1;
}

float deriv_leaky_relu(float a) {
    if (a < 0) return 0.01;
    return 1;
}

float amaxf(float* z, int size) {
    int m = z[0];
    for (int i=0;i<size;i++) {
        if (z[i] > m) m = z[i];
    }
    return m;
}


// Can be more accurate with softmax by computing the Jacobian for backprop
// but would require so much more effort that its not worth it

// Modifies input in place; does not fit typedef
void softmax(float* z, int size) {
    float maxv = amaxf(z, size);
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