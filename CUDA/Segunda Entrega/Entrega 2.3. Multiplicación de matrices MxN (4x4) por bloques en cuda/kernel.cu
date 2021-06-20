#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <stdio.h>  
#include <stdlib.h>
#include <cuda.h> 

__global__ void multi_matrices(int *a, int *b, int *c, int Width) {

	int Pvalue = 0;

	int y = blockIdx.y * Width + threadIdx.y;

	int x = blockIdx.x * Width + threadIdx.x;

	for (int k = 0; k < Width; ++k) {

		int Melement = a[y * Width + k];

		int Nelement = b[k*Width + x];

		Pvalue += Melement * Nelement;
	}

	c[y*Width + x] = Pvalue;
}


int main() {
	const int total_witdh = 16;
	const int Width = 4;
	int a[Width][Width] = { {2,3,4,7},{8,4,1,4},{6,7,4,9},{4,6,1,1} };
	int b[Width][Width] = { {3,4,5,1},{7,8,9,10},{1,3,6,8},{9,8,5,3} };
	int c[Width][Width] = { 0 };
	int* d_c;
	int* d_b;
	int* d_a;

	size_t size = Width * Width * sizeof(int);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

	// Setup the execution configuration+
	//Declaramos que habra un solo grid
	dim3 dimGrid(1, 1);
	//Declaramos que habra 4 bloques de 4 hilos
	dim3 dimBlock(4, 4);

	multi_matrices << < dimGrid, dimBlock >> > (d_a, d_b, d_c, Width);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	//mostramos el resultado por pantalla

	printf("El resultado de la suma es: \n");

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			printf("%d \t", c[i][j]);
		}
		printf("\n");
	}
	printf("\n");

}
