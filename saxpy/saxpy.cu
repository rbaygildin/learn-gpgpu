#include <iostream>
#include "cublas_v2.h"
using namespace std;

#define CUDA_CALL(F, ...)\
	    if((F(__VA_ARGS__)) != cudaSuccess){\
			        cudaError_t e = cudaGetLastError();\
			        printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
			        return(EXIT_FAILURE);\
			    }

const int N = 1 << 10;

int main(){
	float *a_h, *b_h;
	a_h = new float[N];
	b_h = new float[N];
	float *a_d, *b_d;
	for(int i = 0; i < N; i++){
		a_h[i] = (b_h[i] = 1.0f * i);
	}
	cublasHandle_t handle;
	CUDA_CALL(cublasCreate, &handle);
	CUDA_CALL(cudaMalloc, (void**) &a_d, sizeof(float) * N);
	CUDA_CALL(cudaMalloc, (void**) &b_d, sizeof(float) * N);
	CUDA_CALL(cublasSetVector, N, sizeof(float), a_h, 1, a_d, 1);
	CUDA_CALL(cublasSetVector, N, sizeof(float), b_h, 1, b_d, 1);
	const float s = 2.0f;
	CUDA_CALL(cublasSaxpy, handle, N, &s, a_d, 1, b_d, 1);
	CUDA_CALL(cublasGetVector, N, sizeof(float), b_d, 1, b_h, 1);
	CUDA_CALL(cudaFree, a_d);
	CUDA_CALL(cudaFree, b_d);
	CUDA_CALL(cublasDestroy, handle);
	for(int i = 0; i < N; i++)
		cout << "b_h[" << i << "] = " << b_h[i] << endl; 
	delete[] a_h;
	delete[] b_h;
	return 0;
}
