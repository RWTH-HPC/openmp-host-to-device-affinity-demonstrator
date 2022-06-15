#include "memory.hpp"

#include <cuda_runtime.h>

using namespace kernel;

void *memory::pinnedMalloc(size_t size, int const device) {
    void *ptr;

#if (UNIFIED_MEMORY == 0)
    cudaSetDevice(device);
    cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
#elif (UNIFIED_MEMORY == 1)
    cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
#endif
    return ptr;
}

void memory::pinnedFree(void *data) {
    cudaFreeHost(data);
}
