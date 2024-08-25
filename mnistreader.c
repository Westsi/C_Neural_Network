#include "mnistreader.h"
#include <stdint.h>
#include <stdio.h>

FILE* trainingLabels;
FILE* trainingData;
FILE* testLabels;
FILE* testData;

void init() {
    trainingLabels = fopen("./mnist/train-labels", "rb");
    trainingData = fopen("./mnist/train-images", "rb");
    testLabels = fopen("./mnist/t10k-labels", "rb");
    testData = fopen("./mnist/t10k-labels", "rb");
}

uint32_t* readTrainingLabels() {
    fseek(trainingLabels, 0, SEEK_SET);
    unsigned char buf[8];
    fread(buf, sizeof(buf), 1, trainingLabels);
}

uint32_t** readTrainingData() {
    fseek(trainingData, 0, SEEK_SET);
}


uint32_t* readTestLabels() {
    fseek(testLabels, 0, SEEK_SET);
}

uint32_t** readTestData() {
    fseek(testData, 0, SEEK_SET);
}

void closeAll() {
    fclose(trainingLabels);
    fclose(trainingData);
    fclose(testLabels);
    fclose(testData);
}

