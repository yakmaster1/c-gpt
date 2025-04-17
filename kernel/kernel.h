#ifndef KERNEL_H
#define KERNEL_H

void gpu_addDeviceToHostMatrix(float *host_matrix, int host_size, float *device_matrix_pointer);
void gpu_init(float **pointer, float *matrix, int elements);
void kernel_cudafree(float *pointer);
void kernel_memcpydevicetohost(float *dst, float *src, int size);

void gpu_cublas_matmul(float *A, float *B, float *C, int m, int k, int n);

#endif