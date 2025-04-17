#include "kernel.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>

__global__ void vectorAdd_a_to_b(float *cudaA, float *cudaB, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        cudaA[i] += cudaB[i];
    }
    return;
}

void gpu_addDeviceToHostMatrix(float *host_matrix, int host_size, float *device_matrix_pointer)
{
    float *cudaHost = 0;
    gpu_init(&cudaHost, host_matrix, host_size);

    int threads = 256;
    int blocks = (host_size + threads - 1) / threads;
    vectorAdd_a_to_b<<<blocks, threads>>>(cudaHost, device_matrix_pointer, host_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(host_matrix, cudaHost, host_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(cudaHost);
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