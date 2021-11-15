#ifndef GPU_H
#define GPU_H

#include <hwloc.h>

void get_num_numa_nodes(int *num_numa_nodes);
void get_closest_numa_nodes(int num_numa_nodes, int numa_distances[num_numa_nodes][num_numa_nodes]);
void get_current_cpu(int *cpu, int *numa);

int get_hwloc_topology(hwloc_topology_t *topo);
int get_cuda_device_count(int *count);
int get_numa_node_of_gpu(int gpu_device, const hwloc_topology_t *topo, int *numa_node);

int get_closest_gpus(int num_numa_nodes, int num_gpu_devices, int numa_closest_gpus[num_numa_nodes][num_gpu_devices]);

#endif
