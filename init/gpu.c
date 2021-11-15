#define _GNU_SOURCE

#include "gpu.h"

#include <sched.h>
#include <numa.h>
#include <cuda_runtime_api.h>
#include <hwloc/cudart.h>

struct sorting_distance {
    int idx;
    int dist;
};

int sorting_distance_cmp(const void *a, const void *b) {
    return ((struct sorting_distance*)a)->dist - ((struct sorting_distance*)b)->dist;
}

void get_num_numa_nodes(int *num_numa_nodes) {
    if (num_numa_nodes != NULL)
        *num_numa_nodes = numa_num_configured_nodes();
}

void get_closest_numa_nodes(int num_numa_nodes, int numa_distances[num_numa_nodes][num_numa_nodes]) {
    struct sorting_distance distances[num_numa_nodes];
    for (int i = 0; i < num_numa_nodes; i++) {
        for (int j = 0; j < num_numa_nodes; j++) {
            distances[j].idx = j;
            distances[j].dist = numa_distance(i,j);
        }

        qsort(distances, num_numa_nodes, sizeof(struct sorting_distance), sorting_distance_cmp);

        for (int j = 0; j < num_numa_nodes; j++) {
            numa_distances[i][j] = distances[j].idx;
        }
    }
}

void get_current_cpu(int *cpu, int *numa) {
    if (cpu != NULL) {
        *cpu = sched_getcpu();
        if (numa != NULL) {
            *numa = numa_node_of_cpu(*cpu);
        }
    }
}

int get_hwloc_topology(hwloc_topology_t *topo) {
    if (topo == NULL)
        return 1;

    hwloc_topology_init(topo);
    hwloc_topology_set_io_types_filter(*topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(*topo);

    return 0;
}

int get_cuda_device_count(int *count) {
    if (count == NULL)
        return 1;

    return cudaGetDeviceCount(count);
}

int get_numa_node_of_gpu(int gpu_device, const hwloc_topology_t *topo, int *numa_node) {
    if (topo == NULL || numa_node == NULL) {
        return 1;
    }

    hwloc_bitmap_t cpuset;
    cpuset = hwloc_bitmap_alloc();

    hwloc_cudart_get_device_cpuset(*topo, gpu_device, cpuset);
    hwloc_obj_t obj = NULL;
    while (!obj) {
        obj = hwloc_get_next_obj_covering_cpuset_by_type(*topo, cpuset, HWLOC_OBJ_NUMANODE, obj);
    }
    *numa_node = obj->os_index;

    return 0;
}

int get_closest_gpus(
        int num_numa_nodes, 
        int num_gpu_devices, 
        int numa_closest_gpus[num_numa_nodes][num_gpu_devices]) {

    int numa_closest_nodes[num_numa_nodes][num_numa_nodes];
    int numa_gpu[num_numa_nodes][num_gpu_devices];
    int numa_gpu_device_count[num_numa_nodes];
    //int numa_closest_gpu_count[num_numa_nodes];
    hwloc_topology_t topo;

    memset(numa_gpu_device_count, 0, sizeof(*numa_gpu_device_count)*num_numa_nodes);
    //memset(numa_closest_gpu_count, 0, sizeof(*numa_closest_gpu_count)*num_numa_nodes);

    
    if (get_hwloc_topology(&topo)) {
        // Couldn't load topology
        return 1;
    }

    get_closest_numa_nodes(num_numa_nodes, numa_closest_nodes);

    int tmp_numa_node;
    for(int i = 0; i < num_gpu_devices; i++) {
        if (get_numa_node_of_gpu(i, &topo, &tmp_numa_node)) {
            // Couldn't find numa node of gpu device
            return 2;
        }
        numa_gpu[tmp_numa_node][numa_gpu_device_count[i]] = i;
        numa_gpu_device_count[tmp_numa_node]++;
    }

    int tmp_closest_gpu_count;
    for (int i = 0; i < num_numa_nodes; i++) {
        tmp_closest_gpu_count = 0;
        for (int j = 0; j < num_numa_nodes; j++) {
            for (int k = 0; k < numa_gpu_device_count[numa_closest_nodes[i][j]]; k++) {
                numa_closest_gpus[i][tmp_closest_gpu_count] = numa_gpu[numa_closest_nodes[i][j]][k];
                tmp_closest_gpu_count++;
            }
        }
    }

    return 0;
}
