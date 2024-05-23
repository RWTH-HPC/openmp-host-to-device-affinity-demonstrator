#ifndef __KERNEL_MEMORY__
#define __KERNEL_MEMORY__

#include <cstddef>

namespace kernel::memory
{
void *pinnedMalloc(size_t size, int const device);
void pinnedFree(void *data);
} // namespace kernel::memory

#endif // __KERNEL_MEMORY__
