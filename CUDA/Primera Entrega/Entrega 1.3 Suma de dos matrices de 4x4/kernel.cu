#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <stdio.h> 
#include <stdio.h>  
#include <stdlib.h>
#include <cuda.h> 

__global__ void suma_matrices(int *matriz1, int *matriz2, int *final) {
	
	int i= blockIdx.x + threadIdx.x * blockDim.x;

	final[i] = matriz1[i] + matriz2[i];
}


int main() {
	const int N = 4;
	int matriz1[N][N] = { {132,213,22,331},{372,7245,72,722},{2574,222,775,75},{1,25,2,4} };
	int matriz2[N][N] = { {457,225,244,222},{976,257,7456,6467},{5473,543,566,456},{8,365,356,6} };
	int final[N][N] = { 0 };
	int* g_final;
	int* g_matriz1;
	int* g_matriz2;

	size_t size = N * N * sizeof(int);

	cudaMalloc(&g_final, size);
	cudaMalloc(&g_matriz1, size);
	cudaMalloc(&g_matriz2, size);

	cudaMemcpy(g_matriz1, matriz1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(g_matriz2, matriz2, size, cudaMemcpyHostToDevice);
	cudaMemcpy(g_final, final, size, cudaMemcpyHostToDevice);

	suma_matrices <<<4,4>>> (g_matriz1, g_matriz2, g_final);

	cudaMemcpy(final, g_final, size, cudaMemcpyDeviceToHost);

	cudaFree(g_matriz1);
	cudaFree(g_matriz2);
	cudaFree(g_final);

	//mostramos el resultado por pantalla

	printf("El resultado de la suma es: \n");

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%d \t", final[i][j]);
		}
		printf("\n");
	}
	printf("\n");

}
