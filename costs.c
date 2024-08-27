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
        float v = pred[i] + 1e-15;
        sum += actual[i] * logf(v);
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

cost_deriv_t getCostDerivative(cost_func_t costfn) {
    if (costfn == mse) {
        return deriv_mse;
    }
    if (costfn == mae) {
        return deriv_mae;
    }
    if (costfn == binary_cross_entropy) {
        return deriv_binary_cross_entropy;
    }
    if (costfn == categorical_cross_entropy) {
        return deriv_categorical_cross_entropy;
    }
    if (costfn == hinge) {
        return deriv_hinge;
    }
    if (costfn == huber) {
        return deriv_huber;
    }
}

float deriv_mse(float pred, float actual, float outputCnt) {
    float res = (2 / outputCnt) * (pred - actual);
    return res;
}

float deriv_mae(float pred, float actual, float outputCnt) {
    if (pred > actual) return 1.0;
    if (pred == actual) return 0.0001; // return a very very small value. does this need to be changed to 0?
    return -1;
}

// trusting ChatGPT for the below derivatives

float deriv_binary_cross_entropy(float pred, float actual, float outputCnt) {
    float num = pred - actual;
    float denom = pred * (1 - pred);
    return num/denom;
}

// this assumes that actual is a member of a one-hot encoded vector
float deriv_categorical_cross_entropy(float pred, float actual, float outputCnt) {
    if (actual == 1) {
        return pred - 1;
    }
    return pred;
}

float deriv_hinge(float pred, float actual, float outputCnt) {
    float v = 1 - (pred * actual);
    if (v > 0) return -actual;
    return 0;
}

float deriv_huber(float pred, float actual, float outputCnt) {
    float error = actual - pred;
    if (fabsf(error) <= HUBER_DELTA) return -error;
    float sign = 1;
    if (error < 0) sign = -1;
    return -HUBER_DELTA * sign;
}
