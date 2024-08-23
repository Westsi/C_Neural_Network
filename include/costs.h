#ifndef COSTS_H_
#define COSTS_H_

typedef float (*cost_func_t)(float*, float*, int);

#define HUBER_DELTA 1

float mse(float* pred, float* actual, int size);
float mae(float* pred, float* actual, int size);
float binary_cross_entropy(float* pred, float* actual, int size);
float categorical_cross_entropy(float* pred, float* actual, int size);
float hinge(float* pred, float* actual, int size);
float huber(float* pred, float* actual, int size);

typedef float (*cost_deriv_t)(float, float);

cost_deriv_t getCostDerivative(cost_func_t costfn);

float deriv_mse(float pred, float actual);
float deriv_mae(float pred, float actual);
float deriv_binary_cross_entropy(float pred, float actual);
float deriv_categorical_cross_entropy(float pred, float actual);
float deriv_hinge(float pred, float actual);
float deriv_huber(float pred, float actual);

#endif