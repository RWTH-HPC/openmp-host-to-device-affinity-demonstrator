#include <stdlib.h>

#include "kernel.hpp"
#include "../util/define.hpp"

#if (USE_OMP_TARGET == 0)

using namespace kernel;

//static std::vector<cudaStream_t> streams;

__global__ void matrix_mutliply(
        double const *a, double const *b, double *c, int const n) {
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
}

MatrixMultiplyCUDA::MatrixMultiplyCUDA(int device, int num_streams) : MatrixMultiplyDevice(device) {
    cudaSetDevice(device);
    streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStream_t current;
        cudaStreamCreate(&current);
        streams[i] = current;
    }
}

MatrixMultiplyCUDA::~MatrixMultiplyCUDA() {
    free(streams);
}

void MatrixMultiplyCUDA::execute(
        double const *a, double const *b, double *c, int const n) const {
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

void MatrixMultiplyCUDA::executeAsync(
        double const *a, double const *b, double *c, int const n, int const stream_id) const {

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

void MatrixMultiplyCUDA::syncronizeStream(int const stream_id) const {
    cudaStreamSynchronize(streams[stream_id]);
    cudaStreamDestroy(streams[stream_id]);
}
#endif // USE_OMP_TARGET
