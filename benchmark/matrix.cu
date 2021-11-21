#include "matrix.cuh"
#include <vector>

static std::vector<std::vector<void*>> alloc;

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

void kernel::init() {
    int count; 
    cudaGetDeviceCount(&count);
    alloc = std::vector<std::vector<void*>>(count);
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

    #pragma omp critical
    {
        alloc[device].push_back((void*)d_a);
        alloc[device].push_back((void*)d_b);
        alloc[device].push_back((void*)d_c);
    }
}

void kernel::syncronize(const int device) {
    cudaSetDevice(device);
    cudaDeviceSynchronize();
}

void kernel::free(const int device) {
    cudaSetDevice(device);
    for (void* ptr : alloc[device]) {
        cudaFree(ptr);
    }
    alloc[device].clear();
}
