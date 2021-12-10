#include <iostream>
#include <numeric>
#include <random>
#include <chrono>

#include <cuda_runtime_api.h>

#include "matrix.cuh"

int main() {
    const unsigned int N = 2;

    int devices;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&devices);

    for (int i = 0; i < devices; i++) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device: " << i << std::endl << 
            "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl <<
            "Max Blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl <<
            "Multiprocessors: " << prop.multiProcessorCount << std::endl; 
    }


    double a[N*N];
    double b[N*N];
    double c[N*N];
    double c2[N*N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i*N+j] = i*N+j;//rand() / static_cast<double>(RAND_MAX);
            b[i*N+j] = j*N+i;//rand() / static_cast<double>(RAND_MAX);
        }
    }

    kernel::execute_matrix_multiply_kernel(a, b, c, N, 0);
    kernel::execute_matrix_multiply_kernel(b, a, c2, N, 1);

    for (int i = 0; i < N; i++) {
        std::cout << "[ ";
        for (int j = 0; j < N; j++) {
            std::cout << a[i*N+j] << " ";
        }
        std::cout << "] * [ ";
        for (int j = 0; j < N; j++) {
            std::cout << b[i*N+j] << " ";
        }
        std::cout << "] = [ ";
        for (int j = 0; j < N; j++) {
            std::cout << c[i*N+j] << " ";
        }
        std::cout << "]" << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < N; i++) {
        std::cout << "[ ";
        for (int j = 0; j < N; j++) {
            std::cout << b[i*N+j] << " ";
        }
        std::cout << "] * [ ";
        for (int j = 0; j < N; j++) {
            std::cout << a[i*N+j] << " ";
        }
        std::cout << "] = [ ";
        for (int j = 0; j < N; j++) {
            std::cout << c2[i*N+j] << " ";
        }
        std::cout << "]" << std::endl;
    }
}
