#include "costs.h"
#include <math.h>

float mse(float* pred, float* actual, int size) {
    float sum = 0;
    for (int i=0;i<size;i++) {
        sum += powf(pred[i] - actual[i], 2.0);
    }
    float cost = sum / size;
    return cost;
}

float mae(float* pred, float* actual, int size) {
    float sum = 0;
    for (int i=0;i<size;i++) {
        sum += fabsf(pred[i] - actual[i]);
    }
    float cost = sum / size;
    return cost;
}

float binary_cross_entropy(float* pred, float* actual, int size) {
    float sum = 0;

    for (int i=0;i<size;i++) {
        sum += (1 - actual[i]) * logf(1-pred[i] + 1e-7);
        sum += actual[i] * logf(pred[i] + 1e-7);
    }

    return -(sum/size);
}

// ensure that activation fn (softmax) has been called before this
float categorical_cross_entropy(float* pred, float* actual, int size) {
    float sum = 0;
    for (int i=0;i<size;i++) {
        sum += actual[i] * logf(pred[i]);
    }
    return -sum;
}

float hinge(float* pred, float* actual, int size) {
    float sum = 0;
    for (int i=0;i<size;i++) {
        sum += fmaxf(0.0, 1- actual[i] * pred[i]);
    }
    return sum;
}

// DELTA = 1
float huber(float* pred, float* actual, int size) {
    int condition = 1;
    float huber_mse_sum = 0;
    float huber_mae_sum = 0;
    for (int i=0;i<size;i++) {
        float cond = fabsf(actual[i] - pred[i]);
        if (cond > HUBER_DELTA) condition = 0;
        huber_mse_sum += powf(0.5 * (actual[i] - pred[i]), 2.0);
        huber_mae_sum += HUBER_DELTA * (cond - 0.5 * HUBER_DELTA);
    }
    if (condition == 1) {
        return huber_mse_sum;
    }
    return huber_mae_sum;
}