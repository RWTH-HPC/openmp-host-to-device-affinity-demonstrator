// number of tasks
#ifndef NR_TASKS
#define NR_TASKS 200
#endif

#ifndef RANDOMINIT
#define RANDOMINIT 0
#endif

#ifndef RANDOMDIST
#define RANDOMDIST 1
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

#ifndef CHECK_GENERATED_TASK_ID
#define CHECK_GENERATED_TASK_ID 0
#endif

#ifndef SIMULATE_CONST_WORK
#define SIMULATE_CONST_WORK 0
#endif

#ifndef COMPILE_TASKING
#define COMPILE_TASKING 1
#endif

#ifndef USE_TASK_ANNOTATIONS
#define USE_TASK_ANNOTATIONS 0
#endif

#ifndef USE_REPLICATION
#define USE_REPLICATION 0
#endif

#ifndef ITERATIVE_VERSION
#define ITERATIVE_VERSION 1
#endif

#ifndef NUM_ITERATIONS
#define NUM_ITERATIONS 1
#endif

#ifndef NUM_REPETITIONS
#define NUM_REPETITIONS 1
#endif

#ifndef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#if !COMPILE_CHAMELEON
#undef USE_EXTERNAL_CALLBACK
#define USE_EXTERNAL_CALLBACK 0
#endif

#ifndef USE_ALIGNMENT
#define USE_ALIGNMENT 1
#endif

#ifndef USE_HUGE_PAGES
#define USE_HUGE_PAGES 1
#endif

#ifndef GPU
#define GPU 0
#endif

#ifndef COMPUTE
#define COMPUTE 1
#endif

#ifndef ASYNC
#define ASYNC 0
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 2
#endif

#ifndef PINNED_MEMORY
#define PINNED_MEMORY 1
#endif

//#define LOG(rank, str) fprintf(stderr, "#R%d: %s\n", rank, str)
#define LOG(str) printf("%s\n", str)

#define SPEC_RESTRICT __restrict__
//#define SPEC_RESTRICT restrict

#ifndef DPxMOD
#define DPxMOD "0x%0*" PRIxPTR
#endif

#ifndef DPxPTR
#define DPxPTR(ptr) ((int)(2*sizeof(uintptr_t))), ((uintptr_t) (ptr))
#endif
