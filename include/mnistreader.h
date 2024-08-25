#ifndef MNIST_READER_H_
#define MNIST_READER_H_

#include <stdint.h>


void init();
uint32_t* readTrainingLabels();
uint32_t** readTrainingData();

uint32_t* readTestLabels();
uint32_t** readTestData();
void closeAll();

#endif