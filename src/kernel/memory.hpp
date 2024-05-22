#pragma once

#include <cstddef>

namespace kernel::memory {
    void *pinnedMalloc(size_t size, int const device);
    void  pinnedFree(void *data);
}
