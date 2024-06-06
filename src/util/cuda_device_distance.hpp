#ifndef __UTIL_CUDA_DEVICE_DISTANCE__
#define __UTIL_CUDA_DEVICE_DISTANCE__

namespace distance
{
int init();
int get_closest_cuda_devices(const unsigned int numa_node, int desired_number, int *devices);
} // namespace distance

#endif // __UTIL_CUDA_DEVICE_DISTANCE__
