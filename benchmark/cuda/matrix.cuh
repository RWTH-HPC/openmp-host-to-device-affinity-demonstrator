#pragma once

namespace kernel {
    void pin(void *data, size_t size, bool readonly, const int device);
    void unpin(void *data);
    void execute_matrix_multiply_kernel(const double *a, const double *b, double *c, const unsigned int n, const int device);
    void execute_matrix_multiply_kernel_async(const double *a, const double *b, double *c, const unsigned int n, const int device);
    void syncronize(const int device);
    void reset(const int device);
}
