#pragma once

namespace kernel {
    void pin(void *data, size_t size);
    void unpin(void *data);
    void execute_matrix_multiply_kernel(const double *a, const double *b, double *c, const unsigned int n, const int device);
    void syncronize(const int device);
    void reset(const int device);
}
