if (LIBOMPTARGET_NUMA_DEVICE_AFFINITY)
    set(SOURCE_FILES
        kernel.hpp
        omp_kernel.cpp)
else()
    set(SOURCE_FILES
        kernel.hpp
        cuda_kernel.cu)
endif()


if (LIBOMPTARGET_NUMA_DEVICE_AFFINITY)
    # libomptarget does not support static linking of libraries with device code inside
    add_library(distanceBenchmark_kernel SHARED ${SOURCE_FILES})
    set_target_properties(distanceBenchmark_kernel PROPERTIES LINKER_LANGUAGE CXX)
else()
    add_library(distanceBenchmark_kernel ${SOURCE_FILES})
    set_target_properties(distanceBenchmark_kernel PROPERTIES LINKER_LANGUAGE CUDA)
endif()
