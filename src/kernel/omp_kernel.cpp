#include "kernel.hpp"

#if (USE_OMP_TARGET == 1)

using namespace kernel;

MatrixMultiplyOMP::MatrixMultiplyOMP(int device) : MatrixMultiplyDevice(device)
{
}

void MatrixMultiplyOMP::execute(double const *a, double const *b, double *c, size_t const n) const
{
#pragma omp target map(to : a[ : n * n], b[ : n * n]) map(from : c[ : n * n]) device(device)
    {
#if (COMPUTE == 1)
#pragma omp teams distribute parallel for collapse(2)
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                c[i * n + j] = 0;
                for (size_t k = 0; k < n; k++)
                {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
#else
        // To force omp to copy the matrices
        c[0] = a[0] + b[0];
#endif
    }
}

void MatrixMultiplyOMP::executeAsync(double const *a, double const *b, double *c, size_t const n,
                                     int const stream_id) const
{
#pragma omp target map(to : a[ : n * n], b[ : n * n]) map(from : c[ : n * n]) device(device) nowait
    {
#if (COMPUTE == 1)
#pragma omp teams distribute parallel for collapse(2)
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                c[i * n + j] = 0;
                for (size_t k = 0; k < n; k++)
                {
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
                }
            }
        }
#else
        // To force omp to copy the matrices
        c[0] = a[0] + b[0];
#endif
    }
}

#endif // USE_OMP_TARGET
