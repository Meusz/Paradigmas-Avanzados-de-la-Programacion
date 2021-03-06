#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "common\book.h"

int main(void) {
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Global memory on the device: %d\n", prop.totalGlobalMem);
		printf("Maximum amount of shared memory a single block may use : %d\n", prop.sharedMemPerBlock);
		printf("Number of 32 bits registers available er block : %d\n", prop.sharedMemPerBlock);
		printf("Number of threads in a warp : %d\n", prop.warpSize);
		printf("Maximum pitch allowed for memory copies in bytes : %d\n", prop.memPitch);
		printf("Maximum number of threads that a block may contain : %d\n", prop.maxThreadsPerBlock);
		printf("Maximum number of threads allowed along each dimension of a grid : %d\n", prop.maxThreadsDim);
		printf("Number of blocks allowed along each dimension of a grid : %d\n", prop.maxGridSize);
		printf("The amount of available contant memory : %d\n", prop.totalConstMem);
		printf("Major revision of the device compute capability : %d\n", prop.major);
		printf("Minor revision of the device compute capability : %d\n", prop.minor);
		printf("Device`s requirement for texture alligment : %d\n", prop.textureAlignment);
		printf("Number of multiprocessors on device : %d\n", prop.multiProcessorCount);
		printf("Boolean value representing whether there is a limit for kernels executed on this device : %d\n", prop.kernelExecTimeoutEnabled);
		printf("Boolean value representing whether the device is an integrated GPU : %d\n", prop.integrated);
		printf("Boolean value representing whether the device can map into the CUDA device address space : %d\n", prop.canMapHostMemory);
		
		printf("\n");
		getchar();
	}
}