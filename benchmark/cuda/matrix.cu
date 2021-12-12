#include "matrix.cuh"
#include "../util/gpu_distance.hpp"
#include <vector>
#include <omp.h>

__global__ void matrix_mutliply(
        const double *a, const double *b, double *c, const unsigned int n) {

    /*__shared__ double res;

    if (threadIdx.x == 0)
        res = 0;

    double my_val = 
        a[blockIdx.x * n + threadIdx.x] * //a[bx][tx]
        b[threadIdx.x * n + blockIdx.y];  //b[tx][by]

    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
        if (threadIdx.x == i) {
            res += my_val;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        c[blockIdx.x * n + blockIdx.y] = res; //c[bx][by]*/
}

static std::vector<double> dMemoryTime(2);
static std::vector<double> dComputeTime(2);

void kernel::execute_matrix_multiply_kernel(const double *a, 
        const double *b, 
        double *c, 
        const unsigned int n,
        const int device) {

    cudaSetDevice(device);
    dim3 blocks(n,n,1);
    dim3 threads(n,1,1);

    double *d_a;
    double *d_b;
    double *d_c;

    int size = sizeof(double) * n*n;

    double dMemoryStart = omp_get_wtime();

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    double dComputeStart = omp_get_wtime();

    matrix_mutliply<<<blocks,threads>>>(d_a, d_b, d_c, n);
    // Kernels are computed asyncronously
    kernel::syncronize(device);

    double dComputeEnd = omp_get_wtime();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    double dMemoryEnd = omp_get_wtime();

    #pragma omp critical
    {
        dComputeTime[device] += dComputeEnd - dComputeStart;
        dMemoryTime[device] += dMemoryEnd - dMemoryStart - (dComputeEnd - dComputeStart);
    }
}

void kernel::syncronize(const int device) {
    cudaSetDevice(device);
    cudaDeviceSynchronize();
}

double kernel::get_memory_time(const int device) {
    return dMemoryTime[device];
}

double kernel::get_compute_time(const int device) {
    return dComputeTime[device];
}
