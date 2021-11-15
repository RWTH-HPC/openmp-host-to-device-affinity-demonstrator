#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <numa.h>

#include "gpu.h"


int main() {
    double t0 = omp_get_wtime();

    if (numa_available() == -1) {
        printf("NUMA unavailable\n");
        return 1;
    }

    /*printf("0x%x\n", hwloc_get_api_version());
    printf("0x%x\n", HWLOC_API_VERSION);*/

    int num_numa_nodes;
    int num_gpu_devices;

    get_num_numa_nodes(&num_numa_nodes);
    if (get_cuda_device_count(&num_gpu_devices))
        return 1;

    int numa_closest_gpus[num_numa_nodes][num_gpu_devices];
    if(get_closest_gpus(num_numa_nodes, num_gpu_devices, numa_closest_gpus))
        return 1;

    double t1 = omp_get_wtime();
    printf("%f\n", t1- t0);

    #pragma omp parallel
    {
        int curr_cpu, curr_numa_node;
        get_current_cpu(&curr_cpu, &curr_numa_node);
        
        #pragma omp critical
        {
            printf("Thread: %d\tCPU %d\tNUMA node %d\t", omp_get_thread_num(), curr_cpu, curr_numa_node);
            printf("Closest GPUs: [");
            for (int i = 0; i < num_gpu_devices; i++) {
                printf("GPU%d, ", numa_closest_gpus[curr_numa_node][i]);
            }
            printf("]\n");
            fflush(stdout);
        }
    }
    double t2 = omp_get_wtime();
    printf("%f\n", t2- t1);
    return 0;
}
