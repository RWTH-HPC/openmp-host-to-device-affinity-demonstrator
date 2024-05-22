#pragma once

namespace distance {
    int init();
    int get_closest_cuda_device_to_numa_node_by_distance(unsigned int distance_index, const unsigned int numa_node);
}
