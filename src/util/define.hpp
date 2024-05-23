#ifndef __UTIL_DEFINE__
#define __UTIL_DEFINE__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// number of tasks
#ifndef NR_TASKS
#define NR_TASKS 200
#endif

#ifndef PARALLEL_INIT
#define PARALLEL_INIT 1
#endif

#ifndef VERBOSE_MSG
#define VERBOSE_MSG 0
#endif

#ifndef VERBOSE_MATRIX
#define VERBOSE_MATRIX 0
#endif

#ifndef USE_ALIGNMENT
#define USE_ALIGNMENT 1
#endif

#ifndef USE_HUGE_PAGES
#define USE_HUGE_PAGES 0
#endif

#ifndef USE_CLOSEST_GPU
#define USE_CLOSEST_GPU 0
#endif

#ifndef COMPUTE
#define COMPUTE 1
#endif

#ifndef ASYNC
#define ASYNC 0
#endif

// #ifndef BLOCK_SIZE
// #define BLOCK_SIZE 2
// #endif

#ifndef PINNED_MEMORY
#define PINNED_MEMORY 0
#endif

#ifndef UNIFIED_MEMORY
#define UNIFIED_MEMORY 0
#endif

#ifndef USE_OMP_TARGET
#define USE_OMP_TARGET 0
#endif

// #define LOG(rank, str) fprintf(stderr, "#R%d: %s\n", rank, str)
#define LOG(str) printf("%s\n", str)

#define SPEC_RESTRICT __restrict__
// #define SPEC_RESTRICT restrict

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2 * sizeof(uintptr_t))), ((uintptr_t)(ptr))
#endif

#define gpuErrChk(ans)                                                                                                 \
    {                                                                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);                                                                          \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            exit(code);
        }
    }
}

#endif // __UTIL_DEFINE__