#ifndef ACTIVATIONS_H_
#define ACTIVATIONS_H_

#include <math.h>

typedef float (*activation_func_t)(float);

float linear(float z);
float sigmoid(float z);
float logistic(float z);
float a_tanh(float z);
float a_atan(float z);
float relu(float z);
float leaky_relu(float z);

typedef float (*activation_deriv_t)(float);

float deriv_linear(float a);
float deriv_sigmoid(float a);
float deriv_logistic(float a);
float deriv_a_tanh(float a);
float deriv_a_atan(float a);
float deriv_relu(float a);
float deriv_leaky_relu(float a);

activation_deriv_t getActivationDerivative(activation_func_t a);



void softmax(float* z, int size);



#endif