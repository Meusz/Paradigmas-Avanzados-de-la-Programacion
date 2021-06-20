#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "stdio.h"  
#include "stdlib.h"
#include "cuda.h"


#define FILTER_WIDTH 3
#define WIDTH 16
#define CONVULTION_WIDTH WIDTH+2
#define TILE_WIDTH 8

void mostrar_matriz_inicial(int matriz[WIDTH][WIDTH]);
void mostrar_matriz_filtro(int matriz[FILTER_WIDTH][FILTER_WIDTH]);


__global__ void GPU_mediante_un_bloque_y_memoria_global(int *Matriz, int *filtro, int *Salida) {
	//Se definen los ejes x e y de la matriz de entrada

	int y = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int x = blockIdx.x * TILE_WIDTH + threadIdx.x;

	int filtrox = 1;
	int filtroy = 1;

	int Pvalue = 0;
	

	//Necesitamos acceder a una posicion dentro del area permitida
		for (int i = -1; i <= 1; i++)
		{
			if ((y + i) >= WIDTH || (y + i) < 0) {
				Pvalue += 0;
			}
			else {
				for (int j = -1; j <= 1; j++)
				{
					if ((x + j) >= WIDTH || (x + j) < 0) {
						Pvalue += 0;
					}
					else {
						Pvalue += Matriz[ (y + i) * WIDTH + (x + j)] *  filtro[(filtroy + i) * FILTER_WIDTH + (filtrox + j)];
					}
				}
			}
		}
	
	Salida[y * WIDTH + x] = Pvalue;
}



int main() {



	int A[WIDTH][WIDTH] = { 0 };
	int Salida[WIDTH][WIDTH] = { 0 };
	int filtro[FILTER_WIDTH][FILTER_WIDTH] = { 0 };
	int invertida[FILTER_WIDTH][FILTER_WIDTH] = { 0 };
	//int invertida[FILTER_WIDTH][FILTER_WIDTH] = { {1,2,3},{4,5,6},{7,8,9} };
	//Declaramos los bloques dim
	dim3 DimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH);

	//Reservamos la memoria necesaria
	int* d_A;
	int* d_salida;
	int* d_filtro;

	size_t size = WIDTH * WIDTH * sizeof(int);
	size_t size_filtro = FILTER_WIDTH * FILTER_WIDTH * sizeof(int);

	cudaMalloc(&d_A, size);

	cudaMalloc(&d_salida, size);

	cudaMalloc(&d_filtro, size);

	//-------------------------------------------------------- -
		//Rellenamos la matriz A de numeros aleatorios
		//1 + rand() % (99)

		for (int i = 0; i < WIDTH; i++) {
			for (int j = 0; j < WIDTH; j++) {
				A[i][j] = 1 + rand() % (99);
			}
		}
	//Mostramos la matriz inicial
		printf("La matriz A queda como \n: ");

	mostrar_matriz_inicial(A);

	//-------------------------------------------------------- -
		//Rellenamos la matriz Filtro de numeros aleatorios de 0 a 1


		for (int i = 0; i < FILTER_WIDTH; i++) {
			for (int j = 0; j < FILTER_WIDTH; j++) {
				filtro[i][j] = rand() % (2);
			}
		}
	//Mostramos la matriz filtro
		printf("La matriz es filtro queda como \n" );

	mostrar_matriz_filtro(filtro);

	//Calculamos la invertida de la matriz filtro
		int a = 0;
	for (int i = FILTER_WIDTH - 1; i >= 0; i--) {
		int b = 0;
		for (int j = FILTER_WIDTH - 1; j >= 0; j--) {
			invertida[i][j] = filtro[a][b];
			b++;
		}
		a++;
	}
	//Mostramos la matriz filtro invertida
	printf("La invertida de la matriz filtro queda como \n");
	mostrar_matriz_filtro(invertida);


	//Setup the execution configuration +
	//	Cargamos las matrices en la GPU



	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_salida, Salida, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filtro, invertida, size_filtro, cudaMemcpyHostToDevice);


	//Declaramos que habra un solo grid



	GPU_mediante_un_bloque_y_memoria_global  <<< DimGrid, DimBlock >>> (d_A, d_filtro, d_salida);

	//Realizadas las operaciones copiamos los resultados en CPU

	cudaMemcpy(Salida, d_salida, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_salida);
	cudaFree(d_filtro);


	printf("La matriz convulcionada queda como \n");

	mostrar_matriz_inicial(Salida);

	getchar();
}


void mostrar_matriz_inicial(int matriz[WIDTH][WIDTH]) {
	for (int i = 0; i < WIDTH; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d \t", matriz[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void mostrar_matriz_filtro(int matriz[FILTER_WIDTH][FILTER_WIDTH]) {
	for (int i = 0; i < FILTER_WIDTH; i++) {
		for (int j = 0; j < FILTER_WIDTH; j++) {
			printf("%d \t", matriz[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
