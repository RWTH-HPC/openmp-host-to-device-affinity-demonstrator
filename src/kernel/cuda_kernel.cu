#include <stdlib.h>

#include "../util/define.hpp"
#include "kernel.hpp"

#if (USE_OMP_TARGET == 0)

using namespace kernel;

// static std::vector<cudaStream_t> streams;

__global__ void matrix_mutliply(double const *a, double const *b, double *c, size_t const n)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (row < n && col < n)
    {
        // each thread computes one element of the block sub-matrix
        for (size_t i = 0; i < n; i++)
        {
            tmpSum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = tmpSum;
    }
}

MatrixMultiplyCUDA::MatrixMultiplyCUDA(int device, int num_streams) : MatrixMultiplyDevice(device)
{
    gpuErrChk(cudaSetDevice(device));
    streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * num_streams);
    for (int i = 0; i < num_streams; i++)
    {
        cudaStream_t current;
        gpuErrChk(cudaStreamCreate(&current));
        streams[i] = current;
    }
}

MatrixMultiplyCUDA::~MatrixMultiplyCUDA()
{
    free(streams);
}

void MatrixMultiplyCUDA::execute(double const *a, double const *b, double *c, size_t const n) const
{
    gpuErrChk(cudaSetDevice(device));
#if (COMPUTE == 1)
    dim3 threads_per_block(n, n);
    dim3 blocks_per_grid(1, 1);
    if (n * n > 1024)
    {
        threads_per_block.x = 32;
        threads_per_block.y = 32;
        blocks_per_grid.x = (n + threads_per_block.x - 1) / threads_per_block.x;
        blocks_per_grid.y = (n + threads_per_block.y - 1) / threads_per_block.y;
    }
#endif // COMPUTE == 1
    double *d_a;
    double *d_b;
    double *d_c;

    size_t size = sizeof(double) * n * n;

    gpuErrChk(cudaMalloc((void **)&d_a, size));
    gpuErrChk(cudaMalloc((void **)&d_b, size));
    gpuErrChk(cudaMalloc((void **)&d_c, size));

    gpuErrChk(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

#if (COMPUTE == 1)
    matrix_mutliply<<<blocks_per_grid, threads_per_block, 0>>>(d_a, d_b, d_c, n);
#endif // COMPUTE == 1

    gpuErrChk(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

    gpuErrChk(cudaFree(d_a));
    gpuErrChk(cudaFree(d_b));
    gpuErrChk(cudaFree(d_c));
}

void MatrixMultiplyCUDA::executeAsync(double const *a, double const *b, double *c, size_t const n,
                                      int const stream_id) const
{

    gpuErrChk(cudaSetDevice(device));
#if (COMPUTE == 1)
    dim3 threads_per_block(n, n);
    dim3 blocks_per_grid(1, 1);
    if (n * n > 1024)
    {
        threads_per_block.x = 32;
        threads_per_block.y = 32;
        blocks_per_grid.x = (n + threads_per_block.x - 1) / threads_per_block.x;
        blocks_per_grid.y = (n + threads_per_block.y - 1) / threads_per_block.y;
    }
#endif // COMPUTE == 1
    double *d_a;
    double *d_b;
    double *d_c;

    size_t size = sizeof(double) * n * n;

    cudaStream_t stream = streams[stream_id];

    gpuErrChk(cudaMallocAsync((void **)&d_a, size, stream));
    gpuErrChk(cudaMallocAsync((void **)&d_b, size, stream));
    gpuErrChk(cudaMallocAsync((void **)&d_c, size, stream));

    gpuErrChk(cudaMemcpyAsync(d_a, a, size, cudaMemcpyDefault, stream));
    gpuErrChk(cudaMemcpyAsync(d_b, b, size, cudaMemcpyDefault, stream));

#if (COMPUTE == 1)
    matrix_mutliply<<<blocks_per_grid, threads_per_block, 0, stream>>>(d_a, d_b, d_c, n);
#endif // COMPUTE == 1

    gpuErrChk(cudaMemcpyAsync(c, d_c, size, cudaMemcpyDefault, stream));

    gpuErrChk(cudaFreeAsync(d_a, stream));
    gpuErrChk(cudaFreeAsync(d_b, stream));
    gpuErrChk(cudaFreeAsync(d_c, stream));
}

void MatrixMultiplyCUDA::syncronizeStream(int const stream_id) const
{
    gpuErrChk(cudaStreamSynchronize(streams[stream_id]));
    gpuErrChk(cudaStreamDestroy(streams[stream_id]));
}
#endif // USE_OMP_TARGET
