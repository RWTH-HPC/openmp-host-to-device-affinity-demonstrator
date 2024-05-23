#include "memory.hpp"
#include "../util/define.hpp"
#include <cstdio>
#include <cstdlib>
// #include <cuda_runtime.h>

using namespace kernel;

void *memory::pinnedMalloc(size_t size, int const device)
{
    void *ptr;

#if (UNIFIED_MEMORY == 0)
    gpuErrChk(cudaSetDevice(device));
    // printf("cudaSetDevice: err=%d\n", err);
    gpuErrChk(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    // printf("cudaHostAlloc: err=%d\n", err);
#elif (UNIFIED_MEMORY == 1)
    gpuErrChk(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
#endif
    return ptr;
}

void memory::pinnedFree(void *data)
{
    gpuErrChk(cudaFreeHost(data));
}
