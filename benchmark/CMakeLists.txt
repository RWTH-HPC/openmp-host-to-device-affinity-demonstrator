cmake_minimum_required(VERSION 3.13)
project(distanceBenchmark LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(EXTRA_FLAGS "-DCOMPUTE=1 -DASYNC=1 -DPINNED_MEMORY=1")

find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${EXTRA_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${EXTRA_FLAGS}")
set (CMAKE_CUDA_FLAGS "${EXTRA_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_subdirectory(util)
add_subdirectory(cuda)
add_subdirectory(app)

