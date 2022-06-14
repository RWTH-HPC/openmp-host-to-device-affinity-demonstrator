cmake_minimum_required(VERSION 3.13)
project(distanceBenchmark LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(LIBOMPTARGET_NUMA_DEVICE_AFFINITY 1)

set(EXTRA_FLAGS "-DCOMPUTE=1 -DASYNC=1 -DPINNED_MEMORY=0 -DUNIFIED_MEMORY=0")
if (LIBOMPTARGET_NUMA_DEVICE_AFFINITY)
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -DLIBOMPTARGET_NUMA_DEVICE_AFFINITY=1")
else()
    set(EXTRA_FLAGS "${EXTRA_FLAGS} -DLIBOMPTARGET_NUMA_DEVICE_AFFINITY=0")
endif()

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda ${EXTRA_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp-targets=nvptx64-nvidia-cuda ${EXTRA_FLAGS}")
set (CMAKE_CUDA_FLAGS "${EXTRA_FLAGS}")
# Because the environment on claix is broken
if (LIBOMPTARGET_NUMA_DEVICE_AFFINITY)
    set (CMAKE_EXE_LINKER_FLAGS "-L/home/co693196/.local/install/llvm/target-dev/lib ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_subdirectory(util)
add_subdirectory(kernel)
add_subdirectory(app)

