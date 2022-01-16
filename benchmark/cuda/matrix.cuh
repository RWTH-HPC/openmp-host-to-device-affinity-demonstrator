#pragma once

namespace kernel {
    void execute_matrix_multiply_kernel(const double *a, const double *b, double *c, const unsigned int n, const int device);
    void syncronize(const int device);
}
