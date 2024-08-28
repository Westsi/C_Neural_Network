#include <cuda_runtime.h>
#include <stdio.h>

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