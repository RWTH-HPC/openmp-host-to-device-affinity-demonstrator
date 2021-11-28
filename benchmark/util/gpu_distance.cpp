#include <vector>
#include <algorithm>
#include <sched.h>
#include <numa.h>
#include <cuda_runtime_api.h>
#include <hwloc/cudart.h>


#include "gpu_distance.hpp"

#include <iostream>

unsigned int get_numa_nodes_num() {
    return numa_num_configured_nodes();
}

unsigned int get_cuda_device_num() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

void get_current_cpu(unsigned int &cpu, unsigned int &numa) {
    cpu = sched_getcpu();
    numa = numa_node_of_cpu(cpu);
}


void calc_closest_numa_nodes(unsigned int numa_nodes_num, std::vector<std::vector<unsigned int>> &numa_distances) {
    struct index_distance {
        unsigned int index;
        unsigned int distance;
    };

    std::vector<struct index_distance> distances_with_index(numa_nodes_num);
    for (int i = 0; i < numa_nodes_num; i++) {
        for (int j = 0; j < numa_nodes_num; j++) {
            distances_with_index[j].index = j;
            distances_with_index[j].distance = numa_distance(i, j);
        }

        std::sort(distances_with_index.begin(), distances_with_index.end(), 
                [](const struct index_distance &a, const struct index_distance &b) {
                    return a.distance < b.distance;
                });

        for (int j = 0; j < numa_nodes_num; j++) {
            numa_distances[i][j] = distances_with_index[j].index;
        }
    }
}


void get_hwloc_topology(hwloc_topology_t &topo) {
    hwloc_topology_init(&topo);
    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topo);
}

unsigned int get_gpu_numa_node(unsigned int gpu_index, const hwloc_topology_t &topo) {
    hwloc_bitmap_t cpuset;

    cpuset = hwloc_bitmap_alloc();

    hwloc_cudart_get_device_cpuset(topo, gpu_index, cpuset);
    hwloc_obj_t obj = nullptr;
    while (!obj) {
        obj = hwloc_get_next_obj_covering_cpuset_by_type(topo, cpuset, HWLOC_OBJ_NUMANODE, obj);
    }

    unsigned int os_index = obj->os_index;
    free(obj);

    return os_index;
}

void calc_closest_numa_gpus(unsigned int numa_nodes_num, unsigned int cuda_device_num, std::vector<std::vector<unsigned int>> &numa_gpu_lookup_table) {
    std::vector<std::vector<unsigned int>> closest_numa_nodes(numa_nodes_num);
    std::vector<std::vector<unsigned int>> cuda_devices_of_numa_node(numa_nodes_num);

    for (int i = 0; i < numa_nodes_num; i++) {
        closest_numa_nodes[i] = std::vector<unsigned int>(numa_nodes_num);

        cuda_devices_of_numa_node[i].reserve(cuda_device_num);
        numa_gpu_lookup_table[i].reserve(cuda_device_num);
    }


    hwloc_topology_t topo;

    get_hwloc_topology(topo);
    calc_closest_numa_nodes(numa_nodes_num, closest_numa_nodes);

    unsigned int cur_numa_node;
    for (int i = 0; i < cuda_device_num; i++) {
        cur_numa_node = get_gpu_numa_node(i, topo);
        cuda_devices_of_numa_node[cur_numa_node].push_back(i);
    }

    for (int i = 0; i < numa_nodes_num; i++) {
        for (int j = 0; j < numa_nodes_num; j++) {
            cur_numa_node = closest_numa_nodes[i][j];
            for (int k = 0; k < cuda_devices_of_numa_node[cur_numa_node].size(); k++) {
                numa_gpu_lookup_table[i].push_back(cuda_devices_of_numa_node[cur_numa_node][k]);
            }
        }
    }
}


static std::vector<std::vector<unsigned int>> numa_gpu_lookup_table;
static bool initalized = false;


int distance::init() {
    if (numa_available() == -1)
        return -1;

    initalized = true;

    unsigned int cuda_device_num = get_cuda_device_num();
    
    unsigned int numa_nodes_num = get_numa_nodes_num();

    numa_gpu_lookup_table = std::vector<std::vector<unsigned int>>(numa_nodes_num);
    calc_closest_numa_gpus(numa_nodes_num, cuda_device_num, numa_gpu_lookup_table);

    return 0;
}

unsigned int distance::get_gpu_index_by_distance(unsigned int distance_index) {
    if (!initalized) {
        return 0;
    }

    unsigned int cpu, numa_node = 0;
    get_current_cpu(cpu, numa_node);

    if (distance_index >= numa_gpu_lookup_table[numa_node].size())
        distance_index = numa_gpu_lookup_table[numa_node].size() - 1;

    return numa_gpu_lookup_table[numa_node][distance_index];
}
