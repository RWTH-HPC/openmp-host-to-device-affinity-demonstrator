#pragma once

namespace distance {
    int init();
    unsigned int get_gpu_index_by_distance(unsigned int distance_index);
}

unsigned int get_cuda_device_num();
