#include "kernel.hpp"

#if (LIBOMPTARGET_NUMA_DEVICE_AFFINITY == 1)

using namespace kernel;

MatrixMultiplyOMP::MatrixMultiplyOMP(int device) : MatrixMultiplyDevice(device) {}

void MatrixMultiplyOMP::execute(double const *a, double const *b, double *c, int const n) const {
    #pragma omp target teams distribute parallel for \
        map(to:a[:n*n],b[:n*n]) map(from:c[:n*n]) device(device)
#if (COMPUTE == 1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i*n+j] += a[i*n+k]*b[k*n+j];
            }
        }
    }
#else
    // To force omp to copy the matrices
    c[0] = 0;
#endif
}

void MatrixMultiplyOMP::executeAsync(double const *a, double const *b, double *c, int const n, int const stream_id) const {
    #pragma omp target teams distribute parallel for nowait \
        map(to:a[:n*n],b[:n*n]) map(from:c[:n*n]) device(device)
#if (COMPUTE == 1)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                c[i*n+j] += a[i*n+k]*b[k*n+j];
            }
        }
    }
#else
    // To force omp to copy the matrices
    c[0] = 0;
#endif
}

#endif
