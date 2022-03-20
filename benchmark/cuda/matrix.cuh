#pragma once

namespace kernel {
    void pin(void *data, size_t size, bool readonly, const int device);
    void unpin(void *data);
    void *hostPinnedMalloc(size_t size, const int device);
    void hostPinnedFree(void *data);

    void execute_matrix_multiply_kernel(const double *a, const double *b, double *c, const unsigned int n, const int device);
    void execute_matrix_multiply_kernel_async(const double *a, const double *b, double *c, const unsigned int n, const int stream_id, const int device);

    void initStreams(const int num_streams);
    void createStream(const int stream_id, const int device);
    void syncronizeStream(const int stream_id);

    void reset(const int device);
}
