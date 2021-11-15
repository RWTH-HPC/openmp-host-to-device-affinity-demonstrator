#include <iostream>
#include <random>

constexpr const int N = 1024;
constexpr const int M = 32;

__global__ void cu_vector_mul(const float *a, const float *b, float *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    c[index] = a[index] * b[index];
}

int main() {
    float a[N], b[N], c[N];

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
        b[i] = static_cast<float>(rand())/static_cast<float>(RAND_MAX);
        c[i] = 0;
    }

    float *d_a;
    float *d_b;
    float *d_c;


    int size = sizeof(float) * N;

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    cu_vector_mul<<<N/M,M>>>(d_a, d_b, d_c);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);


    for (int i = 0; i < N; i++) {
        std::cout << a[i] << " * " << b[i] << " = " << c[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
