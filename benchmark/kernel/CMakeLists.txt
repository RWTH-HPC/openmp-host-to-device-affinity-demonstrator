if (USE_OMP_TARGET)
    set(SOURCE_FILES
        kernel.hpp
        omp_kernel.cpp
        )
else()
    set(SOURCE_FILES
        kernel.hpp
        cuda_kernel.cu
        )
endif()

# libomptarget does not support static linking of libraries with device code inside
add_library(distanceBenchmark_kernel SHARED ${SOURCE_FILES})

if (USE_OMP_TARGET)
    set_target_properties(distanceBenchmark_kernel PROPERTIES LINKER_LANGUAGE CXX)
else()
    set_target_properties(distanceBenchmark_kernel PROPERTIES LINKER_LANGUAGE CUDA)
endif()


set(SOURCE_FILES
    memory.hpp
    memory.cpp)

find_package(CUDAToolkit REQUIRED)
add_library(distanceBenchmark_kernel_memory ${SOURCE_FILES})
target_link_libraries(distanceBenchmark_kernel_memory PRIVATE CUDA::cudart)
