#include "mnistreader.h"
#include "memtrack.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

FILE* f_trainingLabels;
FILE* f_trainingData;
FILE* f_testLabels;
FILE* f_testData;

void initMnist() {
    f_trainingLabels = fopen("./mnist/train-labels", "rb");
    f_trainingData = fopen("./mnist/train-images", "rb");
    f_testLabels = fopen("./mnist/t10k-labels", "rb");
    f_testData = fopen("./mnist/t10k-images", "rb");
}

float remap(uint8_t data, int min, int max) {
    float r = (float)((float)((float)data - (float)min) / (float)((float)max - (float)min));
    return r;
}

uint32_t swapEndianness(uint8_t* buf) {
    uint32_t r = buf[3] | buf[2] << 8 | buf[1] << 16 | buf[0] << 24;
    return r;
}

uint8_t* readLabels(FILE* labelFile) {
    fseek(labelFile, 0, SEEK_SET);
    uint8_t buf[8];
    fread(buf, sizeof(buf), 1, labelFile);
    // read 8 bytes, holding 2 4 byte integers stored in big endian: magic and size
    uint32_t magic = swapEndianness(buf);
    uint32_t size = swapEndianness(buf+4);
    if (magic != 2049) {
        printf("PANIC Magic number mismatch, expected 2049, got %d\n", magic);
        exit(1);
    }
    uint8_t* labels = malloc(sizeof(uint8_t) * size);
    registerAllocated(labels);
    fread(labels, sizeof(uint8_t), size, labelFile);
    return labels;
}

float** readData(FILE* dataFile) {
    fseek(dataFile, 0, SEEK_SET);
    uint8_t buf[16];
    fread(buf, sizeof(buf), 1, dataFile);
    // read 16 bytes, holding 4 4 byte integers stored in big endian: magic, size, rows and cols
    
    uint32_t magic = swapEndianness(buf);
    uint32_t size = swapEndianness(buf+4);
    uint32_t rows = swapEndianness(buf+8);
    uint32_t cols = swapEndianness(buf+12);

    if (magic != 2051) {
        printf("PANIC Magic number mismatch, expected 2051, got %d\n", magic);
        exit(1);
    }
    printf("magic %d, size, %d, rows %d, cols %d, dsize %d\n", magic, size, rows, cols, rows * cols * size);
    int dsize = rows * cols * size;
    uint8_t* imageData = malloc(sizeof(uint8_t) * dsize);
    fread(imageData, sizeof(uint8_t), dsize, dataFile);
    float** images = malloc(sizeof(float*) * size);
    registerAllocated(images);
    for (int i=0;i<size;i++) {
        float* tempImage = malloc(sizeof(float) * rows * cols);
        registerAllocated(tempImage);
        for (int r=0;r<rows;r++) {
            for (int c=0;c<cols;c++) {
                tempImage[(r * cols) + c] = remap(imageData[(i * rows * cols) + (r * cols) + c], 0, 255);
            }
        }
        images[i] = tempImage;
    }

    free(imageData);
    return images;
}

uint8_t* readTrainingLabels() {
    return readLabels(f_trainingLabels);
}

float** readTrainingData() {
    return readData(f_trainingData);
}


uint8_t* readTestLabels() {
    return readLabels(f_testLabels);
}

float** readTestData() {
    return readData(f_testData);
}

void closeAll() {
    fclose(f_trainingLabels);
    fclose(f_trainingData);
    fclose(f_testLabels);
    fclose(f_testData);
}

