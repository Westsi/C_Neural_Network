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


void softmax(float* z, int size);



#endif