#ifndef KERNEL_H
#define KERNEL_H

void gpu_init(float **pointer, float *matrix, int elements);
void gpu_init_zero(float **pointer, int elements);

void kernel_cudafree(float *pointer);
void kernel_memcpydevicetohost(float *dst, float *src, int size);

void gpu_addPositionMatrix(float *cuda_embed_matrix, float *cuda_position_matrix, int embed_elements);
void gpu_cublas_matmul(float *A, float *B, float *C, int m, int k, int n);

#endif