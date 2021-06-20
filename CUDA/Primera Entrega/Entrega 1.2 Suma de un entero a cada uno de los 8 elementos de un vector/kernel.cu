#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdio.h> #include <stdlib .h>
#include <cuda.h> #include <cuda runtime.h>

__global__ void add2(int ∗a) {
    int i = threadIdx.x;
    a[i] = a[i] + 8;
}

int main() {
    const int N = 8;
    int a[N] = {0,2,43,21,22,45,12,23};
    size_t size = N * sizeof(int);
    int* a_d;
    
    cudaMalloc(&a_d, size);
    cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
    add2<< <1, 8 >> > (a_d);
    cudaMemcpy(a, a_d, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("a = %d\n", a[i]);
    }
    cudaFree(a_d);
}