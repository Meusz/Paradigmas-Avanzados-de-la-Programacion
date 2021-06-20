#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <stdio.h>  
#include <stdlib.h>
#include <cuda.h> 


const int total_witdh = 16;
const int Width = 4;
const int TILE_WIDTH = 2;

__global__ void multi_matrices(int *a, int *b, int *c, int Width) {
	
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;


	// Identify the row and column of
	// the Pd element to work on

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	int Pvalue = 0;
	// Loop over the Md and Nd tiles required
	// to compute the Pd element
	for (int m = 0; m < Width / TILE_WIDTH; ++m) {
		// Collaborative loading of Md and Nd
		// tiles into shared memory
		Mds[ty][tx] = a[Row*Width + (m*TILE_WIDTH + tx)];
		Nds[ty][tx] = b[(m*TILE_WIDTH + ty)*Width + Col];

		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}
	c[Row*Width + Col] = Pvalue;

}


int main() {
	
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
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

	dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);


	multi_matrices << < dimGrid, dimBlock >> > (d_a, d_b, d_c, Width);

	//MATRIZ ORIGINAL
	printf("La matriz original es: \n");

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			printf("%d \t", c[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	//mostramos el resultado por pantalla

	printf("El resultado de la multiplicacion es: \n");

	for (int i = 0; i < Width; i++) {
		for (int j = 0; j < Width; j++) {
			printf("%d \t", c[i][j]);
		}
		printf("\n");
	}
	printf("\n");

}