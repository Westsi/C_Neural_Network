#include <cuda_runtime.h>
#include <stdio.h>

#include "nvcalls.h"

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void checkDevices() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("CUDA Device Count: %d\n", deviceCount);
}

__global__ void multiplyElements(const float *inputs, const float *weights, float *results, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        results[i] = inputs[i] * weights[i];
    }
}

__global__ void reduceSum(const float *input, float *output, int numElements) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + tid;

    sdata[tid] = (i < numElements) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

float computeDotProduct(const float *inputs, const float *weights, int numElements) {
    float *d_inputs, *d_weights, *d_results, *d_partialSums;
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    cudaMalloc((void**)&d_inputs, numElements * sizeof(float));
    cudaMalloc((void**)&d_weights, numElements * sizeof(float));
    cudaMalloc((void**)&d_results, numElements * sizeof(float));
    cudaMalloc((void**)&d_partialSums, numBlocks * sizeof(float));

    cudaMemcpy(d_inputs, inputs, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, numElements * sizeof(float), cudaMemcpyHostToDevice);

    multiplyElements<<<numBlocks, blockSize>>>(d_inputs, d_weights, d_results, numElements);
    reduceSum<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_results, d_partialSums, numElements);

    float *h_partialSums = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        result += h_partialSums[i];
    }

    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_results);
    cudaFree(d_partialSums);
    free(h_partialSums);
    return result;
}