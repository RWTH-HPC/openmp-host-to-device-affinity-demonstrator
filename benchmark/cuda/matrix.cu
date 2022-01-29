#include "../util/define.hpp"

#include "matrix.cuh"
#include <cstddef>
#include <vector>
#include <omp.h>


__global__ void matrix_mutliply(
        const double *a, const double *b, double *c, const unsigned int n) {
    __shared__ double a_block[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ double b_block[BLOCK_SIZE*BLOCK_SIZE];

    //index in the total matrix for which block entry the thread is responsible for
    int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int idy = threadIdx.y + BLOCK_SIZE * blockIdx.y;

    if (idx < n && idy < n) {
        double tmp = 0;
        for (int i = 0; i < n/BLOCK_SIZE; i++) {
            a_block[threadIdx.x + threadIdx.y * BLOCK_SIZE] = a[idy*n + (i*BLOCK_SIZE + threadIdx.x)];
            b_block[threadIdx.x + threadIdx.y * BLOCK_SIZE] = b[(i*BLOCK_SIZE + threadIdx.y)*n + (idx)];
            __syncthreads();

            for (int j = 0; j < BLOCK_SIZE; j++) {
                tmp += a_block[threadIdx.y * BLOCK_SIZE + j] * b_block[j * BLOCK_SIZE + threadIdx.x];
            }
            __syncthreads();
        }
        c[idy*n+idx] = tmp;
    }
}

void kernel::execute_matrix_multiply_kernel(const double *a, 
        const double *b, 
        double *c, 
        const unsigned int n,
        const int device) {

    cudaSetDevice(device);
#if (COMPUTE == 1)
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((n+BLOCK_SIZE-1)/BLOCK_SIZE, (n+BLOCK_SIZE-1)/BLOCK_SIZE);
#endif
    double *d_a;
    double *d_b;
    double *d_c;

    int size = sizeof(double) * n*n;

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);


    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

#if (COMPUTE == 1)
    matrix_mutliply<<<blocks,threads_per_block, 0>>>(d_a, d_b, d_c, n);
#endif

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

void kernel::syncronize(const int device) {
    cudaSetDevice(device);
    cudaDeviceSynchronize();
}

void kernel::pin(void *data, size_t size) {
    cudaHostRegister(data, size, cudaHostRegisterDefault);
}

void kernel::unpin(void *data) {
    cudaHostUnregister(data);
}
