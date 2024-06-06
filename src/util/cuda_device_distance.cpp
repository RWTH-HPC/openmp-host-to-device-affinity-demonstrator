#include "cuda_device_distance.hpp"
#include "system_info.hpp"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <hwloc/cudart.h>
#include <iostream>
#include <numa.h>
#include <vector>

/*
 * Calculates a distance matrix of the numa numa nodes:
 * numa_distances[i][j] = k means numa node k is the j-th closest to numa node i
 */
void get_numa_distances(const unsigned int num_numa_nodes, std::vector<std::vector<unsigned int>> &numa_distances)
{
    struct index_distance
    {
        unsigned int index;
        unsigned int distance;
    };

#ifndef NDEBUG
    std::cout << "DEBUG: Computing NUMA distances" << std::endl;
#endif

    std::vector<struct index_distance> distances_with_index(num_numa_nodes);
    for (int i = 0; i < num_numa_nodes; i++)
    {
        for (int j = 0; j < num_numa_nodes; j++)
        {
            distances_with_index[j].index = j;
            distances_with_index[j].distance = numa_distance(i, j);
        }

        std::sort(
            distances_with_index.begin(), distances_with_index.end(),
            [](const struct index_distance &a, const struct index_distance &b) { return a.distance < b.distance; });

        for (int j = 0; j < num_numa_nodes; j++)
        {
            numa_distances[i][j] = distances_with_index[j].index;
        }
    }
#ifndef NDEBUG
    std::cout << "DEBUG: Computed:" << std::endl;
    for (int i = 0; i < num_numa_nodes; i++)
    {
        std::cout << "DEBUG: Node " << i << ": ";
        for (int j = 0; j < num_numa_nodes; j++)
        {
            std::cout << numa_distances[i][j] << ", ";
        }
        std::cout << std::endl;
    }
#endif
}

/*
 * Returns a hwloc topology object of the system
 */
void get_hwloc_topology(hwloc_topology_t &topo)
{
#ifndef NDEBUG
    std::cout << "DEBUG: Computing HWLOC Topology" << std::endl;
#endif
    hwloc_topology_init(&topo);
    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(topo);
}

/*
 *  Returns the numa node to which the cuda device with index cuda_device_index is connected to
 */
unsigned int get_cuda_device_numa_node(const unsigned int cuda_device_index, const hwloc_topology_t &topo)
{
#ifndef NDEBUG
    std::cout << "DEBUG: Computing the NUMA node of CUDA device GPU" << cuda_device_index << std::endl;
#endif
    hwloc_bitmap_t cpuset;

    cpuset = hwloc_bitmap_alloc();

    hwloc_cudart_get_device_cpuset(topo, cuda_device_index, cpuset);
    hwloc_obj_t obj = nullptr;
    while (!obj)
    {
        obj = hwloc_get_next_obj_covering_cpuset_by_type(topo, cpuset, HWLOC_OBJ_NUMANODE, obj);
    }

    unsigned int os_index = obj->os_index;
    // free(obj);

#ifndef NDEBUG
    std::cout << "DEBUG: Calculated NUMA node " << os_index << std::endl;
#endif

    return os_index;
}

/*
 *  Calculates a matrix representing the closest cuda device of a numa node sorted by numa distance
 *  numa_cuda_device_lookup_table[i][j] = k means the cuda device with index k is the j-th closest device to numa node i
 * by numa distance
 */
void get_cuda_devices_of_numa_node_by_distance(const unsigned int num_numa_nodes, const unsigned int num_cuda_devices,
                                               std::vector<std::vector<unsigned int>> &numa_cuda_device_lookup_table)
{
#ifndef NDEBUG
    std::cout << "DEBUG: Computing CUDA devices lookup table" << std::endl;
#endif
    std::vector<std::vector<unsigned int>> numa_distances(num_numa_nodes);
    std::vector<std::vector<unsigned int>> cuda_devices_of_numa_node(num_numa_nodes);

    for (int i = 0; i < num_numa_nodes; i++)
    {
        numa_distances[i] = std::vector<unsigned int>(num_numa_nodes);

        cuda_devices_of_numa_node[i].reserve(num_cuda_devices);
        numa_cuda_device_lookup_table[i].reserve(num_cuda_devices);
    }

    hwloc_topology_t topo;

    get_hwloc_topology(topo);
    get_numa_distances(num_numa_nodes, numa_distances);

    unsigned int cur_numa_node;
    for (int i = 0; i < num_cuda_devices; i++)
    {
        cur_numa_node = get_cuda_device_numa_node(i, topo);
        cuda_devices_of_numa_node[cur_numa_node].push_back(i);
    }

    for (int i = 0; i < num_numa_nodes; i++)
    {
        for (int j = 0; j < num_numa_nodes; j++)
        {
            cur_numa_node = numa_distances[i][j];
            for (int k = 0; k < cuda_devices_of_numa_node[cur_numa_node].size(); k++)
            {
                numa_cuda_device_lookup_table[i].push_back(cuda_devices_of_numa_node[cur_numa_node][k]);
            }
        }
    }
#ifndef NDEBUG
    std::cout << "DEBUG: Computed:" << std::endl;
    for (int i = 0; i < num_numa_nodes; i++)
    {
        std::cout << "DEBUG: Node " << i << ": ";
        for (int j = 0; j < num_cuda_devices; j++)
        {
            std::cout << "GPU" << numa_cuda_device_lookup_table[i][j] << ", ";
        }
        std::cout << std::endl;
    }
#endif
}

static std::vector<std::vector<unsigned int>> numa_cuda_device_lookup_table;
static bool initalized = false;

/*
 * Initializes the lookup table for cuda devices and gpu nodes
 * Retuns:  0 if successful
 *          -1 if numa is not available
 *          -2 if no cuda devices are available
 */
int distance::init()
{
    if (numa_available() == -1)
        return -1;

    unsigned int num_cuda_devices = system_info::get_num_cuda_devices();
    unsigned int num_numa_nodes = system_info::get_num_numa_nodes();

    if (num_cuda_devices == 0)
        return -2;

    numa_cuda_device_lookup_table = std::vector<std::vector<unsigned int>>(num_numa_nodes);
    get_cuda_devices_of_numa_node_by_distance(num_numa_nodes, num_cuda_devices, numa_cuda_device_lookup_table);

    initalized = true;

    return 0;
}

/*
 * Returns the cuda device indices ordered by closeness to the desired NUMA domain
 * Only returns negative numbers on failure
 */
int distance::get_closest_cuda_devices(const unsigned int numa_node, int desired_number, int *devices)
{
    if (!initalized)
    {
        return -1;
    }

    int dev_found = desired_number;
    if (dev_found > numa_cuda_device_lookup_table[numa_node].size())
        dev_found = numa_cuda_device_lookup_table[numa_node].size();

    for (int i = 0; i < dev_found; i++)
    {
        devices[i] = numa_cuda_device_lookup_table[numa_node][i];
    }
    return dev_found;
}
