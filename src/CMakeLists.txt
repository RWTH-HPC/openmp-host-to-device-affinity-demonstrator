cmake_minimum_required(VERSION 3.13)
project(distanceBenchmark LANGUAGES CXX CUDA)

# set c++17 standard as base
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Additional inclued necessary to build library
include(${CMAKE_SOURCE_DIR}/CMakeModules/OpenMP.cmake)

option(ENABLE_COMPUTE       "Whether or not to enable computation in device kernel" on)
option(ENABLE_ASYNC         "Whether or not to use asynchronous calls" off)
option(ENABLE_PINNED_MEM    "Whether or not to use CUDA pinned memory" off)
option(ENABLE_UNIFIED_MEM   "Whether or not to use CUDA unified memory" off)
option(USE_OMP_TARGET       "Whether or not to use libomptarget. Otherwise CUDA prototype is used" off)
set(LIBOMPTARGET_INSTALL_PATH 0 CACHE STRING "Path to libomptarget lib folder")
set(HWLOC_LOCAL_INSTALL_DIR 0 CACHE STRING "Path to hwloc installation folder")

message(STATUS "===> ENABLE_COMPUTE:           " ${ENABLE_COMPUTE})
message(STATUS "===> ENABLE_ASYNC:             " ${ENABLE_ASYNC})
message(STATUS "===> ENABLE_PINNED_MEM:        " ${ENABLE_PINNED_MEM})
message(STATUS "===> ENABLE_UNIFIED_MEM:       " ${ENABLE_UNIFIED_MEM})
message(STATUS "===> HWLOC_LOCAL_INSTALL_DIR:  " ${HWLOC_LOCAL_INSTALL_DIR})
message(STATUS "===> CMAKE_CUDA_ARCHITECTURES: " ${CMAKE_CUDA_ARCHITECTURES})
message(STATUS "===> USE_OMP_TARGET:           " ${USE_OMP_TARGET})
message(STATUS "===> LIBOMPTARGET_INSTALL_PATH:" ${LIBOMPTARGET_INSTALL_PATH})

set(EXTRA_FLAGS "")
set(EXTRA_DEVICE_FLAGS "")

if (ENABLE_COMPUTE)
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DCOMPUTE=1")
else()
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DCOMPUTE=0")
endif()

if (ENABLE_ASYNC)
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DASYNC=1")
else()
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DASYNC=0")
endif()

if (ENABLE_PINNED_MEM)
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DPINNED_MEMORY=1")
else()
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DPINNED_MEMORY=0")
endif()

if (ENABLE_UNIFIED_MEM)
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DUNIFIED_MEMORY=1")
else()
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DUNIFIED_MEMORY=0")
endif()

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DNDEBUG")
endif()

if (USE_OMP_TARGET)
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DUSE_OMP_TARGET=1")
    set (EXTRA_DEVICE_FLAGS "${EXTRA_DEVICE_FLAGS} -I${LIBOMPTARGET_INSTALL_PATH}/include -fopenmp-targets=nvptx64-nvidia-cuda")
    set (CMAKE_EXE_LINKER_FLAGS "-L${LIBOMPTARGET_INSTALL_PATH}/lib ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    set (EXTRA_FLAGS "${EXTRA_FLAGS} -DUSE_OMP_TARGET=0")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set (CMAKE_C_FLAGS      "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}   ${EXTRA_FLAGS} ${EXTRA_DEVICE_FLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${EXTRA_FLAGS} ${EXTRA_DEVICE_FLAGS}")
set (CMAKE_CUDA_FLAGS   "${EXTRA_FLAGS}")

add_subdirectory(util)
add_subdirectory(kernel)
add_subdirectory(app)

