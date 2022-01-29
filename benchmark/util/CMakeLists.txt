add_library(hwloc_local SHARED IMPORTED)
set_target_properties(hwloc_local PROPERTIES
    IMPORTED_LOCATION "${HWLOC_LOCAL_INSTALL_DIR}/lib/libhwloc.so"
    INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_LOCAL_INSTALL_DIR}/include")

set(SOURCE_FILES
    string.hpp
    cuda_device_distance.hpp
    cuda_device_distance.cpp
    system_info.hpp
    system_info.cpp)

add_library(distanceBenchmark_util ${SOURCE_FILES})
target_link_libraries(distanceBenchmark_util hwloc_local numa)
set_target_properties(distanceBenchmark_util PROPERTIES LINKER_LANGUAGE CXX)
