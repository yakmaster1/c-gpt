#include "kernel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vectorAdd(float *cudaA, float *cudaB, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        cudaA[i] += cudaB[i];
    }
    return;
}

extern "C" void gpu_addvectors(float *matrix_a, float *matrix_b, int size)
{
    float *cudaA = 0;
    float *cudaB = 0;

    size_t bytes = size * sizeof(float);

    cudaMalloc(&cudaA, bytes);
    cudaMalloc(&cudaB, bytes);

    cudaMemcpy(cudaA, matrix_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, matrix_b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(cudaA, cudaB, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(matrix_a, cudaA, bytes, cudaMemcpyDeviceToHost);

    cudaFree(cudaA);
    cudaFree(cudaB);
}