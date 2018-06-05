/*
 ============================================================================
 Name        : cuda_fcm.cu
 Author      : Roman Baygildin
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <random>
#include <ctime>
using namespace std;

#undef DEBUG
#define DEBUG

#define CUDA_CALL(F, ...)\
    if((F(__VA_ARGS__)) != cudaSuccess){\
        cudaError_t e = cudaGetLastError();\
        printf("CUDA failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
        return(EXIT_FAILURE);\
    }

#define PRINT_1D_I(A, S)\
    printf("[");\
    for(int i = 0; i < S; i++){\
        printf("%d, ", A[i]);\
    }\
    printf("]\n");

#define FILL_1D(A, S, V)\
    for(int i = 0; i < S; i++){\
        A[i] = V;\
    }

const int N = 1024;
const int C = 3;
const float M = 2.0f;
const float ERR = 0.001f;

const int MAX_IT = 2;
const int BLOCKDIM = 256;


float* initMu(const int n, const int c){
	random_device rd;
	default_random_engine eng (rd());
	uniform_real_distribution<float> udist(0.0f, 1.0f);
	float *mu = new float[n * c];
	for(int i = 0; i < n; i++){
		float s = 0.0f;
		for(int j = 0; j < c; j++){
			mu[i * c + j] = udist(eng);
			s += mu[i * c + j];
		}
		for(int j = 0; j < c; j++){
			mu[i * c + j] /= s;
		}
	}
	return mu;
}

struct Cluster{
	float xMuS;
	float muS;
};

__global__ void calcDist(const float* src, const float* v, const int n, const int c, float* dist, int it){
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if(pos < n){
		for(int k = 0; k < c; k++){
			dist[pos * c + k] = fabsf(src[pos] - v[k]);
		}
	}
}

#ifndef DEBUG
__global__ void calcMu(const float* src, const float* v, const float* dist, const int n, const int c, const float m, float *mu){
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	float p = 2.0f / (m - 1.0f);
	if(pos < n){
		for(int j = 0; j < c; j++){
			const float currentDist = dist[pos * c + j];
			float newMu = 0.0f;
			for(int k = 0; k < c; k++){
				if(j != k){
					newMu += powf(currentDist / dist[pos * c + k], p);
				}
			}
			mu[pos * c + j] = 1.0f / newMu;

		}
	}
}
#else
__global__ void calcMu(const float* src, const float* v, const float* dist, const int n, const int c, const float m, float *mu, int it){
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	const float p = 2.0f / (m - 1.0f);
	if(pos < n){
		for(int j = 0; j < c; j++){
			const float currentDist = dist[pos * c + j];
			float newMu = 0.0f;
			for(int k = 0; k < c; k++){
				if(j != k){
					newMu += powf(currentDist / dist[pos * c + k], p);
				}
			}
			printf("CALC_MU: it = %d <> NEW_MU[%d, %d] = %f\n", it, pos, j, newMu);
			mu[pos * c + j] = powf(newMu, -1.0f);
		}
	}
}
#endif

#ifdef DEBUG
__global__ void reduceCenters(const float* src, const float* mu, const int n, const int c, const float m, Cluster *reducedV, int it){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	for(int k = 0; k < c; k++){
		_clusters[tid * c + k] = Cluster();
		_clusters[tid * c + k].xMuS = 0.0f;
		_clusters[tid * c + k].muS = 0.0f;
	}
	if(pos < n){
		for(int k = 0; k < c; k++){
			float muInPower = powf(mu[pos * c + k], m);
			_clusters[tid * c + k].xMuS = src[pos] * muInPower;
			_clusters[tid * c + k].muS = muInPower;
//			printf("it = %d <> bid = %d <> tid = %d <> XMUS[%d] = %f\n", it, k, bid, tid, _clusters[tid * c + k].xMuS);
//			printf("it = %d <> bid = %d <> tid = %d <> MUS[%d] = %f\n", it, k, bid, tid, _clusters[tid * c + k].muS);
		}
		__syncthreads();
		for(int k = 0; k < c; k++){
			for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
				if(threadIdx.x < stride){
					_clusters[tid * c + k].xMuS += _clusters[(tid + stride) * c + k].xMuS;
					_clusters[tid * c + k].muS += _clusters[(tid + stride) * c + k].muS;
				}
				__syncthreads();
			}

		}
		__syncthreads();
		if(tid == 0){
			for(int k = 0; k < c; k++){
				reducedV[bid * c + k].xMuS = _clusters[k].xMuS;
				reducedV[bid * c + k].muS = _clusters[k].muS;
			}
		}
	}
}
#else
__global__ void reduceCenters(const float* src, const float* mu, const int n, const int c, const float m, Cluster *reducedV){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	for(int k = 0; k < c; k++){
		_clusters[tid * c + k] = Cluster();
		_clusters[tid * c + k].xMuS = 0.0f;
		_clusters[tid * c + k].muS = 0.0f;
	}
	if(pos < n){
		for(int k = 0; k < c; k++){
			float muInPower = powf(mu[pos * c + k], m);
			_clusters[tid * c + k].xMuS = src[pos] * muInPower;
			_clusters[tid * c + k].muS = muInPower;
		}
		__syncthreads();
		for(int k = 0; k < c; k++){
			for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
				if(threadIdx.x < stride){
					_clusters[tid * c + k].xMuS += _clusters[(tid + stride) * c + k].xMuS;
					_clusters[tid * c + k].muS += _clusters[(tid + stride) * c + k].muS;
				}
				__syncthreads();
			}

		}
		__syncthreads();
		if(tid == 0){
			for(int k = 0; k < c; k++){
				reducedV[bid * c + k].xMuS = _clusters[k].xMuS;
				reducedV[bid * c + k].muS = _clusters[k].muS;
			}
		}
	}
}
#endif

#ifdef DEBUG
__global__ void globalReduceCenters(const Cluster *reducedV, const int n, const int c, float *v, int it){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	if(pos < n){
		for(int k = 0; k < c; k++){
			_clusters[tid * c + k] = reducedV[pos * c + k];
		}
		__syncthreads();
		for(int k = 0; k < c; k++){
			for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
				if(threadIdx.x < stride){
					_clusters[tid * c + k].xMuS += _clusters[(tid + stride) * c + k].xMuS;
					_clusters[tid * c + k].muS += _clusters[(tid + stride) * c + k].muS;
				}
				__syncthreads();
			}
		}
		__syncthreads();
		if(tid == 0){
			for(int k = 0; k < c; k++){
//				printf("it = %d <> XMUS[%d] = %f\n", it, k, _clusters[k].xMuS);
//				printf("it = %d <> MUS[%d] = %f\n", it, k, _clusters[k].muS);
				v[k] = _clusters[k].xMuS / _clusters[k].muS;
//				printf("it = %d <> V[%d] = %f\n", it, k, v[k]);
			}
		}
	}
}

#else
__global__ void globalReduceCenters(const Cluster *reducedV, const int n, const int c, float *v){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	if(pos < n){
		for(int k = 0; k < c; k++){
			_clusters[tid * c + k] = reducedV[pos * c + k];
		}
		__syncthreads();
		for(int k = 0; k < c; k++){
			for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
				if(threadIdx.x < stride){
					_clusters[tid * c + k].xMuS += _clusters[(tid + stride) * c + k].xMuS;
					_clusters[tid * c + k].muS += _clusters[(tid + stride) * c + k].muS;
				}
				__syncthreads();
			}
		}
		__syncthreads();
		if(tid == 0){
			for(int k = 0; k < c; k++){
				v[k] = _clusters[k].xMuS / _clusters[k].muS;
			}
		}
	}
}

#endif

__global__ void defuzzLabels(const float* mu, const int n, const int c, int* labels){
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if(pos < n){
		int index = 0;
		float maxMu = -1.0f;
		for(int k = 0; k < c; k++){
			if(mu[pos * c + k] > maxMu){
				index = k;
				maxMu = mu[pos * c + k];
			}
		}
		labels[pos] = index;
	}
}

int main(int argc, const char** argv){
	//Host variables
	float *src = new float[N];
	int* labels = new int[N];
	float* mu = initMu(N, C);

	//Fill src by random values
	random_device rd;
	default_random_engine eng (rd());
	uniform_real_distribution<float> udist(0.0f, 1.0f);
	FILL_1D(src, N, udist(eng));

	//Num of iterations
	int it = 0;

	//Device variables
	float *src_d = new float[N];
	int* labels_d = new int[N];
	float *mu_d = new float[N * C];
	float *v_d = new float[C];
	float *dist_d = new float[N * C];
	Cluster *reduced_v = new Cluster[((BLOCKDIM + N - 1) / BLOCKDIM)];

	//Allocated device variables
	CUDA_CALL(cudaMalloc, (void**) &src_d, sizeof(float) * N);
	CUDA_CALL(cudaMalloc, (void**) &labels_d, sizeof(int) * N);
	CUDA_CALL(cudaMalloc, (void**) &mu_d, sizeof(float) * N * C);
	CUDA_CALL(cudaMalloc, (void**) &v_d, sizeof(float) * C);
	CUDA_CALL(cudaMalloc, (void**) &dist_d, sizeof(float) * N * C);
	CUDA_CALL(cudaMalloc, (void**) &reduced_v, sizeof(float) * BLOCKDIM * sizeof(struct Cluster) * C)

	//Copy from host to device
	CUDA_CALL(cudaMemcpy, src_d, src, sizeof(float) * N, cudaMemcpyHostToDevice);
	CUDA_CALL(cudaMemcpy, mu_d, mu, sizeof(float) * N * C, cudaMemcpyHostToDevice);

	//Size of shared memory in reduction
	const int sharedSize1 = BLOCKDIM * sizeof(struct Cluster) * C;
	const int sharedSize2 = ((BLOCKDIM + N - 1) / BLOCKDIM) * sizeof(struct Cluster) * C;

	//Run Fuzzy C Means
	it = 1;
	do{
#ifndef DEBUG
		reduceCenters<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM, sharedSize1>>>(src_d, mu_d, N, C, M, reduced_v);
		globalReduceCenters<<<1, (BLOCKDIM + N - 1) / BLOCKDIM, sharedSize2>>>(reduced_v, N, C, v_d);
		calcDist<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM>>>(src_d, mu_d, N, C, dist_d);
		calcMu<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM>>>(src_d, v_d, dist_d, N, C, M, mu_d);
#else
		reduceCenters<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM, sharedSize1>>>(src_d, mu_d, N, C, M, reduced_v, it);
		globalReduceCenters<<<1, (BLOCKDIM + N - 1) / BLOCKDIM, sharedSize2>>>(reduced_v, N, C, v_d, it);
		calcDist<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM>>>(src_d, mu_d, N, C, dist_d, it);
		calcMu<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM>>>(src_d, v_d, dist_d, N, C, M, mu_d, it);
#endif


	}while(++it <= MAX_IT);

	//Defuzzification of labels
	defuzzLabels<<<(BLOCKDIM + N - 1) / BLOCKDIM, BLOCKDIM>>>(mu_d, N, C, labels_d);

	//Copy from device to host
	CUDA_CALL(cudaMemcpy, labels, labels_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

	PRINT_1D_I(labels, N);

	//Clean device memory
	CUDA_CALL(cudaFree, src_d);
	CUDA_CALL(cudaFree, labels_d);
	CUDA_CALL(cudaFree, mu_d);
	CUDA_CALL(cudaFree, v_d);
	CUDA_CALL(cudaFree, dist_d);

	//Clean host memory
	delete[] src;
	delete[] labels;
	delete[] mu;
	return 0;
}
