#include "../util/define.hpp"

#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <inttypes.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include "math.h"
#include <malloc.h>
#include <cmath>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <atomic>
#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include <omp.h>

#include "../util/string.hpp"
#include "../util/cuda_device_distance.hpp"
#include "../util/system_info.hpp"
#include "../cuda/matrix.cuh"

#if CHECK_GENERATED_TASK_ID
#include <mutex>
#endif

typedef enum matrix_size_mode_t {
    matrix_size_mode_normal = 0,
    matrix_size_mode_non_uniform = 1
} matrix_size_mode_t;

matrix_size_mode_t matrix_size_mode = matrix_size_mode_normal;
int numberOfTasks = 0;

// mode: normal
int matrixSize = 100;

// mode: non-uniform
typedef enum non_uniform_ordering_t {
    non_uniform_ordering_high_to_low = 0,
    non_uniform_ordering_low_to_high = 1
} non_uniform_ordering_t;

typedef struct non_uniform_matrix_settings_t {
    int matrix_size;
    int number_tasks;
} non_uniform_matrix_settings_t;

non_uniform_ordering_t non_uniform_ordering = non_uniform_ordering_high_to_low;
std::vector<non_uniform_matrix_settings_t> non_uniform_matrix_settings;
std::vector<int> non_uniform_full_array_matrix_sizes;

#define MEM_ALIGNMENT 4096

static inline void* alloc(size_t size)
{
#if USE_ALIGNMENT
    void* p = memalign(MEM_ALIGNMENT, size);
#else
    void* p = malloc(size);
#endif
#if !USE_HUGE_PAGES
    madvise(p, size, MADV_NOHUGEPAGE);
#endif
    return p;
}

void initialize_matrix_rnd(double *mat, int matrixSize) {
    double lower_bound = 0;
    double upper_bound = 10000;
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;

    for(int i=0; i<matrixSize*matrixSize; i++) {
        mat[i]= unif(re);
    }
}

void initialize_matrix_zero(double *mat, int matrixSize) {
    for(int i=0; i<matrixSize*matrixSize; i++) {
        mat[i]= 0;
    }
}

void initialize_matrix_test_A(double *mat, int matrixSize) {
    for(int i=0; i<matrixSize*matrixSize; i++) {
            mat[i]= 1;
    }
}

bool check_test_matrix(double *c, int matrix_idx, double val, int matrixSize) {
    if (NUM_REPETITIONS > 0) {
        for(int i=0;i<matrixSize;i++) {
            for(int j=0;j<matrixSize;j++) {
                if(fabs(c[i*matrixSize+j] - val) > 1e-3) {
                    printf("(OS_TID:%ld): Error in matrix %03d entry (%d,%d) expected:%f but value is %f\n", syscall(SYS_gettid),matrix_idx,i,j,val,c[i*matrixSize+j]);
                    return false;
                }
            }
        }
    }
    return true;
}

void compute_random_task_distribution(int *dist, int nRanks) {
    double *weights = new double[nRanks];

    double lower_bound = 0;
    double upper_bound = 1;
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::default_random_engine re;
    double sum = 0;

    for(int i=0; i<nRanks; i++) {
        weights[i]= unif(re);
        sum += weights[i];
    }

    for(int i=0; i<nRanks; i++) {
        weights[i]= weights[i]/sum;
        dist[i] = weights[i]*NR_TASKS;
    }

    delete[] weights;
}

void printHelpMessage() {
    std::cout << "Usage (mode=normal): ./matrixExample matrixSize [nt_(0) ... nt_(np-1)] " << std::endl;
    std::cout << "    Arguments: " << std::endl;
    std::cout << "        matrixSize:   Number of elements of the matrixSize x matrixSize matrices" << std::endl;
    std::cout << "        nt_(i):       Number of tasks for process i " << std::endl;
    std::cout << "    If the number of tasks is not specified for every process, the application will generate an initial task distribution" << std::endl << std::endl;

    std::cout << "Usage (mode=non-uniform): ./matrixExample non-uniform matrixSizes numberTasks [order_(0) ... order_(np-1)] " << std::endl;
    std::cout << "    Arguments: " << std::endl;
    std::cout << "        matrixSizes:  Comma separated list of different matrix sizes for non-uniform task creation" << std::endl;
    std::cout << "        numberTasks:  Comma separated list defining number of tasks for each matrix size" << std::endl;
    std::cout << "        order_(i):    Ordering of tasks using matrix sizes for rank/process i; 0=\"high to low\" (default); 1=\"low to high\"" << std::endl << std::endl;
}

int parse_command_line_args(int argc, char **argv) {
    if(argc>=2 && strcmp(argv[1], "non-uniform")==0) {
        matrix_size_mode = matrix_size_mode_non_uniform;

        if(argc < 4) {
            std::cout << "Error: Insufficient number parameters" << std::endl;
            printHelpMessage();
            return 1;
        }

        // parse matrix sizes and number of tasks
        std::string str_msizes(argv[2]);
        std::list<std::string> cur_split_msizes = split(str_msizes, ',');
        std::string str_ntasks(argv[3]);
        std::list<std::string> cur_split_ntasks = split(str_ntasks, ',');
        if(cur_split_msizes.size() != cur_split_ntasks.size()) {
            std::cout << "Error: Number of matrix sizes and number of tasks does not match!" << std::endl;
            return 1;
        }

        for (std::string s : cur_split_msizes) {
            non_uniform_matrix_settings_t new_obj;
            new_obj.matrix_size = std::atoi(s.c_str());
            non_uniform_matrix_settings.push_back(new_obj);
        }

        numberOfTasks = 0;
        int count = 0;
        for (std::string s : cur_split_ntasks) {
            int tmp_num = std::atoi(s.c_str());
            non_uniform_matrix_settings[count].number_tasks = tmp_num;
            numberOfTasks += tmp_num;
            count++;
        }

        // parse ordering
        if(argc > 4) {
            if(argc != 5) {
                std::cout << "Error: Number of matrix ordering values does not match number of processes/ranks!" << std::endl;
                return 1;
            }
            int tmp_order = std::atoi(argv[4]);
            non_uniform_ordering = (non_uniform_ordering_t)tmp_order;
        }

        // apply ordering
        if (non_uniform_ordering == non_uniform_ordering_high_to_low) {
            std::sort(
                non_uniform_matrix_settings.begin(),
                non_uniform_matrix_settings.end(),
                [](const non_uniform_matrix_settings_t & a, const non_uniform_matrix_settings_t & b) -> bool
                {
                    return a.matrix_size > b.matrix_size;
                }
            );
        } else {
            std::sort(
                non_uniform_matrix_settings.begin(),
                non_uniform_matrix_settings.end(),
                [](const non_uniform_matrix_settings_t & a, const non_uniform_matrix_settings_t & b) -> bool
                {
                    return b.matrix_size > a.matrix_size;
                }
            );
        }

        non_uniform_full_array_matrix_sizes.clear();
        for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
            for (int i = 0; i < s.number_tasks; i++) {
                non_uniform_full_array_matrix_sizes.push_back(s.matrix_size);
            }
        }

        // ===== DEBUG
        // printf("Rank#%d - Ordering: %d\n", my_rank_id, non_uniform_ordering);
        // for (non_uniform_matrix_settings_t s : non_uniform_matrix_settings) {
        //     printf("Rank#%d - MatrixSize: %d, NumTasks: %d\n", my_rank_id, s.matrix_size, s.number_tasks);
        // }
        // printf("Rank#%d - Size Array: ", my_rank_id);
        // for (int s : non_uniform_full_array_matrix_sizes) {
        //     printf("%d,", s);
        // }
        // printf("\n");
        // ===== DEBUG

    } else if(argc==2) {
        matrix_size_mode = matrix_size_mode_normal;
        matrixSize = atoi( argv[1] );
        if(RANDOMDIST) {
            int *dist = new int[1];
            compute_random_task_distribution(dist, 1);
            delete[] dist;
        } else {
            numberOfTasks = NR_TASKS;
        }
    } else if(argc==3) {
        matrix_size_mode = matrix_size_mode_normal;
        LOG("using user-defined initial load distribution...");
        matrixSize = atoi( argv[1] );
        numberOfTasks = atoi( argv[2] );
    } else {
        printHelpMessage();
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    double fTimeStart, fTimeEnd;
    double wTimeCham, wTimeHost;
    bool pass = true;

    int ret_code = parse_command_line_args(argc, argv);
    if (ret_code != 0) {
        return ret_code;
    }

    if(matrix_size_mode == matrix_size_mode_normal) {
        printf("Mode: Normal Task Distribution\n");
    } else if (matrix_size_mode == matrix_size_mode_non_uniform) {
        printf("Mode: Non-Uniform Task Distribution\n");
    }

#if (COMPUTE == 1)
    LOG("Computation activated, memory and computation performance is measured");
#elif (COMPUTE == 0)
    LOG("Computation deactivated, only memory performance is measured");
#endif

    std::cout << "Will create "+std::to_string(numberOfTasks)+" tasks" << std::endl;
    if (numberOfTasks % omp_get_max_threads() != 0)
        std::cout << "Warning: Number of tasks not evenly dividable by number of threads, threads will have different workloads" << std::endl;

    // GPU distance initalization
    fTimeStart = omp_get_wtime();
    int err = distance::init();
    fTimeEnd = omp_get_wtime();
    if (err == -1) {
        std::cout << "Error: NUMA is not available on this system" << std::endl;
        return -1;
    } else if (err == -2) {
        std::cout << "Error: No CUDA devices where found on this system" << std::endl;
        return -2;
    }
    std::cout << "CUDA device distance initalization was successful and took " << fTimeEnd-fTimeStart << std::endl;

    unsigned int num_cuda_devices = system_info::get_num_cuda_devices();
    std::cout << "Num CUDA devices found: " << num_cuda_devices << std::endl;
    system_info::get_cuda_device_info(0);

    std::vector<double> thread_waiting_time(omp_get_max_threads()*32);
    std::vector<unsigned int> thread_gpu(omp_get_max_threads()*32);


#if (GPU == 0)
    unsigned int target_distance_index = 0;
#elif (GPU == 1)
    unsigned int target_distance_index = num_cuda_devices - 1;
#endif

    #pragma omp parallel
    {
        unsigned int cpu, numa;
        system_info::get_current_cpu(cpu, numa);
        thread_gpu[omp_get_thread_num()*32] = distance::get_closest_cuda_device_to_numa_node_by_distance(target_distance_index, numa);
    }


    double **matrices_a, **matrices_b, **matrices_c;
    matrices_a = new double*[numberOfTasks];
    matrices_b = new double*[numberOfTasks];
    matrices_c = new double*[numberOfTasks];

    printf("Executing parallel init\n");
    fTimeStart=omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(int i=0; i<numberOfTasks; i++) {

        int cur_size = matrixSize;
        if(matrix_size_mode == matrix_size_mode_non_uniform) {
            cur_size = non_uniform_full_array_matrix_sizes[i];
        }
#if (PINNED_MEMORY == 0)
        matrices_a[i] = (double*) alloc((long)cur_size*cur_size*sizeof(double));
        matrices_b[i] = (double*) alloc((long)cur_size*cur_size*sizeof(double));
        matrices_c[i] = (double*) alloc((long)cur_size*cur_size*sizeof(double));
#else
        matrices_a[i] = (double*) kernel::hostPinnedMalloc((size_t)cur_size*cur_size*sizeof(double), thread_gpu[omp_get_thread_num()*32]);
        matrices_b[i] = (double*) kernel::hostPinnedMalloc((size_t)cur_size*cur_size*sizeof(double), thread_gpu[omp_get_thread_num()*32]);
        matrices_c[i] = (double*) kernel::hostPinnedMalloc((size_t)cur_size*cur_size*sizeof(double), thread_gpu[omp_get_thread_num()*32]);
#endif
        if(RANDOMINIT) {
            initialize_matrix_rnd(matrices_a[i], cur_size);
            initialize_matrix_rnd(matrices_b[i], cur_size);
            initialize_matrix_zero(matrices_c[i], cur_size);
        }
        else {
            initialize_matrix_test_A(matrices_a[i], cur_size);
            initialize_matrix_test_A(matrices_b[i], cur_size);
            initialize_matrix_zero(matrices_c[i], cur_size);
        }
    }
    double memory_allocation_time = omp_get_wtime()-fTimeStart;
    std::cout << "Memory Allocation duration: " << memory_allocation_time << std::endl;


    fTimeStart=omp_get_wtime();
#if (ASYNC == 1)
    kernel::initStreams(omp_get_max_threads());
    for (int i = 0; i < omp_get_max_threads(); i++)
        kernel::createStream(i, thread_gpu[i*32]);
#endif

    #pragma omp parallel for schedule(static)
    for(int i=0; i<numberOfTasks; i++) {
        int cur_size = matrixSize;
        if(matrix_size_mode == matrix_size_mode_non_uniform) {
            cur_size = non_uniform_full_array_matrix_sizes[i];
        }

        // Currently not necessary to create tasks and make sure that 
        // iteration is really executed by desired thread
        // #pragma omp task default(shared) firstprivate(i,cur_size)
        // {
            double t0 = omp_get_wtime();
#if (ASYNC == 0)
            kernel::execute_matrix_multiply_kernel(
                    matrices_a[i], matrices_b[i], matrices_c[i], cur_size, 
                    thread_gpu[omp_get_thread_num()*32]);
#elif (ASYNC == 1)
            kernel::execute_matrix_multiply_kernel_async(
                    matrices_a[i], matrices_b[i], matrices_c[i], cur_size, 
                    omp_get_thread_num(),
                    thread_gpu[omp_get_thread_num()*32]);
#endif
            thread_waiting_time[omp_get_thread_num()*32] += omp_get_wtime() - t0;
        // }
    }

#if (ASYNC == 1)
    for (int i = 0; i < omp_get_max_threads(); i++)
        kernel::syncronizeStream(i);
#endif
    fTimeEnd=omp_get_wtime();
    wTimeHost = fTimeEnd-fTimeStart;

    printf("Computations with normal tasking took %.5f\n", wTimeHost);
    for (int i = 0; i < omp_get_max_threads(); i++) {
        std::cout << "Invocation latency of thread " << i << " on GPU" << thread_gpu[i*32] << ": " << thread_waiting_time[i*32] << std::endl;
    }
    // TODO: fix min max calculation when working with padded array
    const auto [min, max] = std::minmax_element(thread_waiting_time.begin(), thread_waiting_time.end());
    std::cout << "Ratio longest waiting time / shortest waiting time: "  << *max / *min << std::endl;


#if (COMPUTE == 1)
    pass = true;
    if(numberOfTasks>0) {
        for(int t=0; t<numberOfTasks; t++) {
            int cur_size = matrixSize;
            if(matrix_size_mode == matrix_size_mode_non_uniform) {
                cur_size = non_uniform_full_array_matrix_sizes[t];
            }

            /*for (int i = 0; i < cur_size; i++) {
                std::cout << "[ ";
                for (int j = 0; j < cur_size; j++) {
                    std::cout << matrices_a[t][i*cur_size+j] << " ";
                }
                std::cout << "] * [ ";
                for (int j = 0; j < cur_size; j++) {
                    std::cout << matrices_b[t][i*cur_size+j] << " ";
                }
                std::cout << "] = [ ";
                for (int j = 0; j < cur_size; j++) {
                    std::cout << matrices_c[t][i*cur_size+j] << " ";
                }
                std::cout << "]" << std::endl;
            }*/

            pass &= check_test_matrix(matrices_c[t], t, cur_size, cur_size);
        }
        if(pass)
            LOG("Validation: TEST SUCCESS");
        else
            LOG("Validation: TEST FAILED");
    }
#elif (COMPUTE == 0)
    LOG("Validation skipped");
#endif


    //deallocate matrices
    for(int i=0; i<numberOfTasks; i++) {
#if (PINNED_MEMORY == 0)
        free(matrices_a[i]);
        free(matrices_b[i]);
        free(matrices_c[i]);
#else
        kernel::hostPinnedFree(matrices_a[i]);
        kernel::hostPinnedFree(matrices_b[i]);
        kernel::hostPinnedFree(matrices_c[i]);
#endif 
    }

    delete[] matrices_a;
    delete[] matrices_b;
    delete[] matrices_c;

    return 0;
}
