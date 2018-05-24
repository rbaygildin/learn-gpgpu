// -*- mode: C -*-
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

#define FILL(A, V, S) \
	for(size_t i = 0; i < S; i++){\
		A[i] = V;\
	}\

#define SUM(ACC, A, S) \
	for(size_t i = 0; i < S; i++){\
		ACC += A[i]; \
	}\

__global__ void vecAdd(float* a, float *b, float *res, int n) { 
	int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id < n){
    res[id] = a[id] + b[id];
  }
}

int main(int argc, const char** argv) {

  float *a, *b, *c;
  size_t size = sizeof(float) * N;
  a = (float*) malloc(size);
  b = (float*) malloc(size);
  c = (float*) malloc(size);
  FILL(a, 1.0, N);
  FILL(b, 2.0, N);

  float *deviceA, *deviceB, *deviceC;
  cudaMalloc((void**) &deviceA, size);
  cudaMalloc((void**) &deviceB, size);
  cudaMalloc((void**) &deviceC, size);

  cudaMemcpy(deviceA, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, b, size, cudaMemcpyHostToDevice);

  size_t gridSize = N >> 1;
  size_t blockSize = N >> 1;
  vecAdd <<<gridSize, blockSize>>> (deviceA, deviceB, deviceC, N);

  cudaMemcpy(c, deviceC, size, cudaMemcpyDeviceToHost);
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  float sum = 0.0;
  SUM(sum, c, N);

  printf("Sum = %f\n", sum);
  printf("Assert = %f\n", sum / 3.0 / N);
  free(a);
  free(b);
  free(c);
  return 0;  
}