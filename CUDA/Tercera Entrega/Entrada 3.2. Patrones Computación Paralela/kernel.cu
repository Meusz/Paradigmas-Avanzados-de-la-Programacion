#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <stdio.h>  
#include <stdlib.h>
#include <cuda.h> 

#define WIDTH 16
#define TILE_WIDTH 8

__device__ int minimo_multiplo(int num) {
	int i = 2;
	while (num % i != 0) {
		i++;
	}
	return i;
}

__global__ void Stencil(int *a, int *c) {
	//Se suman todos los numeros adyacentes al de la casilla central. Los adyacentes se ponen a 0

	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

	if (a[y*WIDTH + x] == 9) {
		int Pvalue = a[y*WIDTH + x];
		
		c[(y - 1)*WIDTH + x] = 0;
		Pvalue += a[(y - 1)*WIDTH + x];
		c[(y + 1)*WIDTH + x] = 0;
		Pvalue += a[(y + 1)*WIDTH + x];
		c[y*WIDTH +(x - 1)] = 0;
		Pvalue += a[y*WIDTH + (x - 1)];
		c[y*WIDTH + (x + 1)] = 0;
		Pvalue += a[y*WIDTH + (x + 1)];

		c[y*WIDTH + x] = Pvalue;
	}
	else {
		c[y*WIDTH + x] = a[y*WIDTH + x];
	}
	
}
__global__ void Scatter(int *a, int *c) {
	//Si el numero es 11, suma 11 a todos los numeros en su columna
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

	if (a[y*WIDTH + x] == 11) {

		for (int i = 0; (y - i) >= 0 ; i++) {
			c[(y - i)*WIDTH + x ] = a[(y - i)*WIDTH + x] + 11;
		}
		for (int i = 0; (y + i) < WIDTH; i++)
		{
			c[(y + i)*WIDTH + x ] = a[(y + i)*WIDTH + x ] + 11;
		}

		c[y*WIDTH + x] = 11;
		
				
	}
	else 
	{
		c[y*WIDTH + x] = a[y*WIDTH + x];
	}

}
__global__ void Gather(int *a, int *c) {
	//Si hay una fila con 3 o mas numeros con el minimo comun multiplo, la posicion que mas en el medio esta acumula la suma de los
	//numeros y los demas asumen valor 0

	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int comun = minimo_multiplo(a[y*WIDTH + x]);
	int izq = 0, der = 0;
	while (comun == minimo_multiplo(a[y*WIDTH + x + der]) ) {
		der++;
	}
	while (x - izq>=0 && comun == minimo_multiplo(a[y*WIDTH + x - izq]) ) {
		izq++;
	}
	izq--; der--;
	if ( (izq - der) ==1 && izq + der>1 ) {
		int Pvalue = 0;
		for (int i = x - izq; i <= x + der; i++)
		{
			Pvalue += a[y*WIDTH + i];
			c[y*WIDTH + i] = 0;
		}
		c[y*WIDTH + x] = Pvalue;
	}
	else {
		c[y*WIDTH + x] = a[y*WIDTH + x];
	}
}

/*__global__ void Gatherf(int *a, int *c) {
	//Si hay una fila con 3 numeros con un minimo comun multiplo, la posicion que mas en el medio esta acumula la suma de los
	//numeros y los demas asumen valor 0
	
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int comun = minimo_multiplo(a[y*WIDTH + x]);
	int izq = 0, der = 0;
	while ( x + der < WIDTH && comun == minimo_multiplo(a[y*WIDTH + x + der])) {
		der++;
	}
	while (x - izq >= 0 && comun == minimo_multiplo(a[y*WIDTH + x - izq])) {
		izq++;
	}
	izq--; der--;
	if ( izq + der > 2) {
		if (( (izq - der) == 1 || (izq - der) == 0)) {
			int Pvalue = 0;
			for (int i = 1; i <= der; i++)
			{
				Pvalue += a[y*WIDTH + x+der];
			}
			for (int i = 1; i <= izq; i++)
			{
				Pvalue += a[y*WIDTH + x -izq];
			}
			c[y*WIDTH + x] = Pvalue + a[y*WIDTH + x];
			
			
		}
		else {
		c[y*WIDTH + x] = 0;
		}
	}
	else {
		c[y*WIDTH + x] += a[y*WIDTH + x];
	}
}
__global__ void Gatherc(int *a, int *c) {
	//Si hay una columna con 3 o mas numeros con el minimo comun multiplo, la posicion que mas en el medio esta acumula la suma de los
	//numeros y los demas asumen valor 0
	int Pvalue = 0;
	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int comun = minimo_multiplo(a[y*WIDTH + x]);
	int abj = 0, ari = 0;
	while (y + ari < WIDTH && comun == minimo_multiplo(a[(y+ ari+1 )*WIDTH + x])) {
		ari++;
	}
	while (y - abj  >= 0 && comun == minimo_multiplo(a[(y- abj-1 )*WIDTH + x])) {
		abj++;
	}
	if (abj + ari > 2) {
		if (((abj - ari) == 1 || (abj - ari) == 0)) {
			c[y*WIDTH + x] = c[y*WIDTH + x] + a[y*WIDTH + x];


		}
		else {
			int diff = ari - abj;
			if ((ari + abj) % 2 != 0 && diff % 2 != 0) {
				diff += 1;

			}
			diff = diff / 2;

			c[y*WIDTH + x] = 0;

			c[(y + diff)*WIDTH + x] = c[(y + diff)*WIDTH + x] + a[y*WIDTH + x];



		}

	}
	else {
		c[y*WIDTH + x] = a[y*WIDTH + x];
	}
}*/


int main() {
	
	
	int a[WIDTH][WIDTH] = { 0 };
	int c[WIDTH][WIDTH] = { 0 };
	int* d_c;
	int* d_b;
	int* d_a;
	int* d_d;
	dim3 DimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH);

	size_t size = WIDTH * WIDTH * sizeof(int);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_c, size);
	
	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/

	//Rellenamos las matrices a y b de numeros aleatorios 
	//1 + rand() % (99)
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			a[i][j] = 1 + rand() % (99);
		}
	}
	//mostramos el resultado por pantalla
	printf("La matriz inicial es: \n");

	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d \t", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/


	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

	// Setup the execution configuration+
	//Declaramos que habra un solo grid
	
	

	Stencil << < DimGrid, DimBlock >> > (d_a, d_c);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);
	cudaFree(d_c);
	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/
	printf("El resultado de stencil es: \n");
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d \t", c[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/

	int b[WIDTH][WIDTH] = { 0 };

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);

	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


	Scatter << < DimGrid, DimBlock >> > (d_a, d_b);

	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/
	printf("El resultado de scatter es: \n");
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d \t", b[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/
	
	cudaFree(d_a);
	cudaFree(d_b);

	/*--------------------------------------------------------------*/
	int d[WIDTH][WIDTH] = { 0 };

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_d, size);
	
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_d, d, size, cudaMemcpyHostToDevice);


	Gather << < DimGrid, DimBlock >> > (d_a, d_b);

	cudaMemcpy(d, d_d, size, cudaMemcpyDeviceToHost);

	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/
	printf("El resultado de Gatherc es: \n");
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d \t", d[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	/*---------------------------------------------------------
	-----------------------------------------------------------
	-----------------------------------------------------------*/

	cudaFree(d_a);
	cudaFree(d_d);
}