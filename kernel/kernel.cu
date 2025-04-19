#include "kernel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

#define MAX_THREADS 256

__global__ void vectorAdd(float *cudaA, float *cudaB, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        cudaA[i] += cudaB[i];
    }
    return;
}

void gpu_addPositionMatrix(float *cuda_embed_matrix, float *cuda_position_matrix, int embed_elements)
{
    int threads = 256;
    int blocks = (embed_elements + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(cuda_embed_matrix, cuda_position_matrix, embed_elements);
    cudaDeviceSynchronize();
    return;
}

// m = rows of A
// k = cols of A = rows of B
// n = cols of B
// C = result
void gpu_cublas_matmul(float *A, float *B, float *C, int m, int k, int n)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        A, m,
        B, k,
        &beta,
        C, m
    );

    cublasDestroy(handle);
    return;
}

void gpu_init_zero(float **pointer, int elements)
{
    float *zero = (float*)calloc(elements, sizeof(float));
    if(zero == NULL) exit(EXIT_FAILURE);
    gpu_init(pointer, zero, elements);
    free(zero);
    return;
}

void gpu_init(float **pointer, float *matrix, int elements)
{
    size_t bytes = elements * sizeof(float);
    cudaMalloc(pointer, bytes);
    cudaMemcpy(*pointer, matrix, bytes, cudaMemcpyHostToDevice);
    return;
}

void kernel_cudafree(float *pointer) 
{
    cudaFree(pointer);
    return;
}

void kernel_memcpydevicetohost(float *dst, float *src, int size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    return;
}