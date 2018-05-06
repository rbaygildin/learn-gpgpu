#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls like __popc and whatever.
#define __CUDACC__

#include <device_functions.h>

// These headers are all implicitly present when you compile CUDA with clang. Clion doesn't know that, so
// we include them explicitly to make the indexer happy. Doing this when you actually build is, obviously,
// a terrible idea :D
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_cmath.h>

#endif // __JETBRAINS_IDE__

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>

#define CUDA_CALL(F, ...)\
    if((F(__VA_ARGS__)) != cudaSuccess){\
        cudaError_t e = cudaGetLastError();\
        printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
        return(EXIT_FAILURE);\
    }

#define CURAND_CALL(F, ...)\
    if((F(__VA_ARGS__)) != CURAND_STATUS_SUCCESS){\
        cudaError_t e = cudaGetLastError();\
        if(e != cudaSuccess){\
            printf("CuRAND failure %s:%d: '%s'\n",__FILE__,__LINE__, cudaGetErrorString(e));\
        }\
        return(EXIT_FAILURE);\
    }

#define PRINT_1D(A, S)\
    printf("[");\
    for(int i = 0; i < S; i++){\
        printf("%f, ", A[i]);\
    }\
    printf("]\n");

#define PRINT_FLAT2D(A, WIDTH, HEIGHT)\
    printf("[\n");\
    for(int i = 0; i < WIDTH; i++){\
        printf("[");\
        for(int j = 0; j < HEIGHT; j++){\
            printf("%f, ", A[i + j * WIDTH]);\
        }\
        printf("]\n");\
    }\
    printf("]\n");

#define W 20
#define H 20
#define N (W * H)
#define BLOCKDIM 1024
#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32

__global__ void seq(float* dst, int w, int h){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	dst[tid] = tid;
}

__global__ void memsetVarF32(float *src, float c){
	*src = c;
}

__global__ void sum(const float* src, float *s, int n){
	__shared__ float _s[BLOCKDIM]; 
	int tid = threadIdx.x;
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if(id < n){
		_s[tid] = src[id];
	}
	else{
		_s[tid] = 0.0;
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
		if(tid < stride){
			_s[tid] += _s[tid + stride];
		}
	}
	__syncthreads();
	if(tid == 0){
		atomicAdd(s, _s[0]);
	}
}

int main(){
	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate, &start);
	CUDA_CALL(cudaEventCreate, &stop);
	float *src;
	float *devSrc;
	float *devAvg;
	float *devS;
	float _avg = 0.0;
	src = (float*) malloc(N * sizeof(float));
	CUDA_CALL(cudaMalloc, (void**) &devSrc, N * sizeof(float));
	CUDA_CALL(cudaMalloc, (void**) &devAvg, sizeof(float));
	CUDA_CALL(cudaMalloc, (void**) &devS, sizeof(float));

	dim3 blockSize(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 gridSize((W + BLOCKDIM_X) / W, (H + BLOCKDIM_Y) / H);
	seq<<<(N + BLOCKDIM) / BLOCKDIM, BLOCKDIM>>>(devSrc, W, H);
	// memsetVarF32<<<1, 1>>>(devS, 0.0);

	cudaEventRecord(start);
	memsetVarF32<<<1, 1>>>(devS, 0.0);
	sum<<<(N + BLOCKDIM) / BLOCKDIM, BLOCKDIM>>>(devSrc, devS, N);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	CUDA_CALL(cudaMemcpy, src, devSrc, N * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CALL(cudaMemcpy, &_avg, devS, sizeof(float), cudaMemcpyDeviceToHost);
	// PRINT_1D(src, N);
	printf("AVG = %f\n", _avg);
	printf("TIME = %f (msec)\n", milliseconds);
	CUDA_CALL(cudaFree, devSrc);
	CUDA_CALL(cudaFree, devAvg);
	CUDA_CALL(cudaFree, devS);
	CUDA_CALL(cudaEventDestroy, start);
	CUDA_CALL(cudaEventDestroy, stop);
	free(src);
	return EXIT_SUCCESS;
}