set(SOURCE_FILES
    matrix_example_api.cpp)

add_executable(distanceBenchmark_app ${SOURCE_FILES})
add_dependencies(distanceBenchmark_app distanceBenchmark_util distanceBenchmark_cuda)
target_link_libraries(distanceBenchmark_app distanceBenchmark_util distanceBenchmark_cuda)

