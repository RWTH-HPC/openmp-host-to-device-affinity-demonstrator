#include "../util/define.hpp"

#include "matrix.cuh"
#include <cstddef>
#include <vector>
#include <omp.h>

static std::vector<cudaStream_t> streams;

__global__ void matrix_mutliply(
        const double *a, const double *b, double *c, const unsigned int n) {
    //__shared__ double a_block[BLOCK_SIZE*BLOCK_SIZE];
    //__shared__ double b_block[BLOCK_SIZE*BLOCK_SIZE];

    //int tasks_per_thread = (BLOCK_SIZE*BLOCK_SIZE + blockDim.x*blockDim.y - 1)/(blockDim.x*blockDim.y);

    //int *idx = new int[tasks_per_thread];
    //int *idy = new int[tasks_per_thread];
    //double *tmp = new double[tasks_per_thread];

    //for (int task = 0; task < tasks_per_thread; task++) {
    //    int tx = ((threadIdx.x + threadIdx.y * BLOCK_SIZE) * tasks_per_thread + task) % BLOCK_SIZE;
    //    int ty = ((threadIdx.x + threadIdx.y * BLOCK_SIZE) * tasks_per_thread + task) / BLOCK_SIZE;

    //    idx[task] = tx + BLOCK_SIZE * blockIdx.x;
    //    idy[task] = ty + BLOCK_SIZE * blockIdx.y;

    //    tmp[task] = 0;
    //}


    //for (int i = 0; i < (n+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
    //    for (int task = 0; task < tasks_per_thread; task++) {
    //        int tx = ((threadIdx.x + threadIdx.y * BLOCK_SIZE) * tasks_per_thread + task) % BLOCK_SIZE;
    //        int ty = ((threadIdx.x + threadIdx.y * BLOCK_SIZE) * tasks_per_thread + task) / BLOCK_SIZE;

    //        if (i*BLOCK_SIZE + tx < n)
    //            a_block[tx + ty * BLOCK_SIZE] = a[idy[task]*n + (i*BLOCK_SIZE + tx)];
    //        else
    //            a_block[tx + ty * BLOCK_SIZE] = 0;

    //        if (i*BLOCK_SIZE + ty < n)
    //            b_block[tx + ty * BLOCK_SIZE] = b[(i*BLOCK_SIZE + ty)*n + (idx[task])];
    //        else
    //            b_block[tx + ty * BLOCK_SIZE] = 0;
    //    }

    //    __syncthreads();

    //    for (int task = 0; task < tasks_per_thread; task++) {
    //        int tx = ((threadIdx.x + threadIdx.y * BLOCK_SIZE) * tasks_per_thread + task) % BLOCK_SIZE;

    //        for (int j = 0; j < BLOCK_SIZE; j++) {
    //            tmp[task] += a_block[tx * BLOCK_SIZE + j] * b_block[j * BLOCK_SIZE + tx];
    //        }
    //    }
    //    __syncthreads();
    //}
    //for (int task = 0; task < tasks_per_thread; task++) {
    //    if (idx[task] < n && idy[task] < n) {
    //        c[idy[task]*n+idx[task]] = tmp[task];
    //    }
    //}

    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (row < n && col < n) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < n; i++) {
            tmpSum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = tmpSum;
    }

    //int64_t cycles = 0;
    //int64_t start = clock64();
    //while(cycles < 1480500 * 100) {
    //    cycles = clock64() - start;
    //}


}

void kernel::execute_matrix_multiply_kernel(const double *a, 
        const double *b, 
        double *c, 
        const unsigned int n,
        const int device) {

    cudaSetDevice(device);
#if (COMPUTE == 1)
    dim3 threads_per_block(n, n);
    dim3 blocks_per_grid(1, 1);
    if (n*n > 1024){
        threads_per_block.x = 32;
        threads_per_block.y = 32;
        blocks_per_grid.x = (n + threads_per_block.x - 1) / threads_per_block.x;
        blocks_per_grid.y = (n + threads_per_block.y - 1) / threads_per_block.y;
    }
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
    matrix_mutliply<<<blocks_per_grid,threads_per_block, 0>>>(d_a, d_b, d_c, n);
#endif

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

void kernel::execute_matrix_multiply_kernel_async(const double *a, 
        const double *b, 
        double *c, 
        const unsigned int n,
        const int stream_id,
        const int device) {

    cudaSetDevice(device);
#if (COMPUTE == 1)
    dim3 threads_per_block(n, n);
    dim3 blocks_per_grid(1, 1);
    if (n*n > 1024){
        threads_per_block.x = 32;
        threads_per_block.y = 32;
        blocks_per_grid.x = (n + threads_per_block.x - 1) / threads_per_block.x;
        blocks_per_grid.y = (n + threads_per_block.y - 1) / threads_per_block.y;
    }
#endif
    double *d_a;
    double *d_b;
    double *d_c;

    int size = sizeof(double) * n*n;

    cudaStream_t stream = streams[stream_id];

    cudaMallocAsync((void **)&d_a, size, stream);
    cudaMallocAsync((void **)&d_b, size, stream);
    cudaMallocAsync((void **)&d_c, size, stream);


    cudaMemcpyAsync(d_a, a, size, cudaMemcpyDefault, stream);
    cudaMemcpyAsync(d_b, b, size, cudaMemcpyDefault, stream);

#if (COMPUTE == 1)
    matrix_mutliply<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_a, d_b, d_c, n);
#endif

    cudaMemcpyAsync(c, d_c, size, cudaMemcpyDefault, stream);

    cudaFreeAsync(d_a, stream);
    cudaFreeAsync(d_b, stream);
    cudaFreeAsync(d_c, stream);
}

void kernel::initStreams(const int num_streams) {
    streams = std::vector<cudaStream_t>(num_streams);
}

void kernel::createStream(const int stream_id, const int device) {
    cudaSetDevice(device);

    cudaStream_t cur;
    cudaStreamCreate(&cur);
    streams[stream_id] = cur;
}

void kernel::syncronizeStream(const int stream_id) {
    cudaStreamSynchronize(streams[stream_id]);
    cudaStreamDestroy(streams[stream_id]);
}

void kernel::pin(void *data, size_t size, bool readonly, const int device) {
    cudaSetDevice(device);
    if (readonly)
        cudaHostRegister(data, size, cudaHostRegisterReadOnly);
    else
        cudaHostRegister(data, size, cudaHostRegisterDefault);
}

void kernel::unpin(void *data) {
    cudaHostUnregister(data);
}

void *kernel::hostPinnedMalloc(size_t size, const int device) {
    void *ptr;

#if (UNIFIED_MEMORY == 0)
    cudaSetDevice(device);
    cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
#elif (UNIFIED_MEMORY == 1)
    cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
#endif
    return ptr;
}

void kernel::hostPinnedFree(void *data) {
    cudaFreeHost(data);
}
