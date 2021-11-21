#include "matrix.cuh"
#include <vector>

__global__ void matrix_mutliply(
        const double *a, const double *b, double *c) {
    __shared__ double res;

    if (threadIdx.x == 0)
        res = 0;

    double my_val = 
        a[blockIdx.x * blockDim.x + threadIdx.x] * //a[bx][tx]
        b[threadIdx.x * blockDim.x + blockIdx.y];  //b[tx][by]

    __syncthreads();

    for (int i = 0; i < blockDim.x; i++) {
        if (threadIdx.x == i) {
            res += my_val;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        c[blockIdx.x * blockDim.x + blockIdx.y] = res; //c[bx][by]
}

void kernel::execute_matrix_multiply_kernel_async(const double *a, 
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

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    matrix_mutliply<<<blocks,threads>>>(d_a, d_b, d_c);
    cudaMemcpyAsync(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFreeAsync(d_a, 0);
    cudaFreeAsync(d_b, 0);
    cudaFreeAsync(d_c, 0);
}

void kernel::syncronize(const int device) {
    cudaSetDevice(device);
    cudaDeviceSynchronize();
}
