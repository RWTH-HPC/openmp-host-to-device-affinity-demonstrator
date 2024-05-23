#ifndef __UTIL_SYSTEM_INFO__
#define __UTIL_SYSTEM_INFO__

namespace system_info
{
unsigned int get_num_cuda_devices();
unsigned int get_num_numa_nodes();
void get_current_cpu(unsigned int &cpu, unsigned int &numa);
void get_cuda_device_info(unsigned int device);
} // namespace system_info

#endif // __UTIL_SYSTEM_INFO__
