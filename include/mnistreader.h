#ifndef MNIST_READER_H_
#define MNIST_READER_H_

#include <stdint.h>


void initMnist();
uint8_t* readTrainingLabels();
float** readTrainingData();

uint8_t* readTestLabels();
float** readTestData();
void closeAll();

#endif