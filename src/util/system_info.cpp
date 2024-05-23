#include "system_info.hpp"
#include "define.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <numa.h>
#include <sched.h>

/*
 * Returns how many cuda capable computing devices are available
 */
unsigned int system_info::get_num_cuda_devices()
{
    int count;
    gpuErrChk(cudaGetDeviceCount(&count));
    return count;
}

/*
 * Returns how many numa nodes the system uses
 */
unsigned int system_info::get_num_numa_nodes()
{
    return numa_num_configured_nodes();
}

/*
 * Get the cpu and numa node the current thread is running on
 */
void system_info::get_current_cpu(unsigned int &cpu, unsigned int &numa)
{
    cpu = sched_getcpu();
    numa = numa_node_of_cpu(cpu);
}

void system_info::get_cuda_device_info(unsigned int device)
{
    cudaDeviceProp props;
    gpuErrChk(cudaGetDeviceProperties(&props, device));

    std::cout << "asyncEngineCount " << props.asyncEngineCount << std::endl;
    std::cout << "clockRate " << props.clockRate << std::endl;
    std::cout << "concurrentKernels " << props.concurrentKernels << std::endl;
    std::cout << "maxBlocksPerMultiProcessor " << props.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerBlock " << props.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "multiProcessorCount " << props.multiProcessorCount << std::endl;
    std::cout << "name[256] " << props.name << std::endl;
    std::cout << "unifiedAddressing " << props.unifiedAddressing << std::endl;
}
