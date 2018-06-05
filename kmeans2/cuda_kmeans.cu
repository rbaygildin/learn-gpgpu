#include <assert.h>
#include <cstdio>
#include <random>

using namespace std;


#define MAX_IT 1
#define N (1024 * 1024)
#define N_CLUSTERS 6
#define BLOCKDIM 1024
#define BLOCKDIM2 ((N + BLOCKDIM - 1) / BLOCKDIM)

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

#define PRINT_1D_I(A, S)\
    printf("[");\
    for(int i = 0; i < S; i++){\
        printf("%d, ", A[i]);\
    }\
    printf("]\n");

#define PRINT_1D_F(A, S)\
    printf("[");\
    for(int i = 0; i < S; i++){\
        printf("%f, ", A[i]);\
    }\
    printf("]\n");

#define FILL_1D(A, S, V)\
    for(int i = 0; i < S; i++){\
        A[i] = V;\
    }

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

struct Cluster{
	float sum;
	int count;

	__device__ inline void operator += (Cluster& a){
		this->count += a.count;
		this->sum += a.sum;
	}
};

texture<float, 1, cudaReadModeElementType> tex;

__device__ Cluster clusters_d[N_CLUSTERS * ((N + BLOCKDIM - 1) / BLOCKDIM)];

__device__ float euclidianDist(const float a, const float b){
	return fabsf(a - b);
}

__global__ void relabel_k(const float* clusters, int* labels){
	extern __shared__ Cluster _clusters[];
	int tid = threadIdx.x;
	int pos = threadIdx.x + blockIdx.x * blockDim.x;

#pragma unroll
	for(unsigned int c = 0; c < N_CLUSTERS; c++){
		Cluster cluster;
		cluster.sum = 0.0f;
		cluster.count = 0;
		_clusters[N_CLUSTERS * tid + c] = cluster;
	}
	__syncthreads();
	if(pos < N){
		float minDist = 1.0f;
		int clusterIndex = 0;
		float val = tex1Dfetch(tex, pos);
#pragma unroll
		for(int c = 0; c < N_CLUSTERS; c++){
			float dist = euclidianDist(val, clusters[c]);
			if(dist <= minDist){
				clusterIndex = c;
				minDist = dist;
			}
		}
		labels[pos] = clusterIndex;
		_clusters[N_CLUSTERS * tid + clusterIndex].sum = val;
		_clusters[N_CLUSTERS * tid + clusterIndex].count = 1;
	}
	__syncthreads();

#pragma unroll
	for(unsigned int c = 0; c < N_CLUSTERS; c += 2){
#pragma unroll
		for(unsigned int stride = BLOCKDIM >> 1; stride > 0; stride >>= 1){
			if(tid < stride){
				_clusters[N_CLUSTERS * tid + c] += _clusters[N_CLUSTERS * (tid + stride) + c];
			}
			else if(c + 1 < N_CLUSTERS){
				_clusters[N_CLUSTERS * (BLOCKDIM - tid - 1) + c + 1] += _clusters[N_CLUSTERS * (BLOCKDIM - tid + stride - 1) + c + 1];
			}
			__syncthreads();
		}
	}
	if(tid == 0){
#pragma unroll
		for(unsigned int c = 0; c < N_CLUSTERS; c++){
			clusters_d[N_CLUSTERS * blockIdx.x + c] = _clusters[c];
		}
	}
}

//__global__ void calculateClusters_k(const int* labels, const int clusterIndex){
//	extern __shared__ Cluster _clusters[];
//	int pos = threadIdx.x + blockIdx.x * blockDim.x;
//	int tid = threadIdx.x;
//	_clusters[tid] = Cluster();
//	_clusters[tid].sum = 0.0f;
//	_clusters[tid].count = 0;
//	if(pos < N && labels[pos] == clusterIndex){
//		_clusters[tid].sum = tex1Dfetch(tex, pos);
//		_clusters[tid].count = 1;
//	}
//	__syncthreads();
//	for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
//		if(threadIdx.x < stride){
//			_clusters[tid].sum += _clusters[tid + stride].sum;
//			_clusters[tid].count += _clusters[tid + stride].count;
//		}
//	__syncthreads();
//	}
//
//	if(tid == 0){
//		clusters_d[blockIdx.x].sum = _clusters[0].sum;
//		clusters_d[blockIdx.x].count = _clusters[0].count;
////		printf("BlockIDX = %d, Sum = %f, Count = %d", blockIdx.x, _clusters[0].sum, _clusters[0].count);
//	}
//}

__global__ void findCenters_k(float* newCenters){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockIdx.x * BLOCKDIM2;
	int tid = threadIdx.x;
	int bid = blockIdx.x;

#pragma unroll
	for(unsigned int c = 0; c < N_CLUSTERS; c++){
		_clusters[N_CLUSTERS * tid + c] = clusters_d[tid + bid * N_CLUSTERS * BLOCKDIM2];
	}
	__syncthreads();

#pragma unroll
	for(unsigned int c = 0; c < N_CLUSTERS; c += 2){

#pragma unroll
		for(unsigned int stride = BLOCKDIM2 >> 1; stride > 0; stride >>= 1){
			if(tid < stride){
				_clusters[N_CLUSTERS * tid + c] += _clusters[N_CLUSTERS * (tid + stride) + c];
			}
			else if(c + 1 < N_CLUSTERS){
				_clusters[N_CLUSTERS * (BLOCKDIM2 - tid - 1) + c + 1] += _clusters[N_CLUSTERS * (BLOCKDIM2 - tid - 1 + stride) + c + 1];
			}
			__syncthreads();
		}
	}

	if(tid == 0){

#pragma unroll
		for(unsigned int c = 0; c < N_CLUSTERS; c++){
			newCenters[c] = _clusters[c].count > 0 ? fdividef(_clusters[c].sum, _clusters[c].count) : 0.0f;
		}
	}
}

int main(){
    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    CUDA_CALL(cudaMemGetInfo, &free_byte, &total_byte);
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024 / 1024, free_db / 1024 / 1024, total_db / 1024 / 1024);

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0.0, 1.0);


	const int blockSize = (N + BLOCKDIM - 1) / BLOCKDIM;
	float* src = new float[N];
	int* labels = new int[N];
	float* centers = new float[N_CLUSTERS];

	float* src_d, *centers_d;
//	Cluster* clusters_d;
	int* labels_d;

	CUDA_CALL(cudaMalloc, (void**)&src_d, sizeof(float) * N);
	CUDA_CALL(cudaMalloc, (void**)&labels_d, sizeof(int) * N);
	CUDA_CALL(cudaMalloc, (void**)&centers_d, sizeof(float) * N_CLUSTERS);
//	CUDA_CALL(cudaMalloc, (void**)&clusters_d, sizeof(struct Cluster) * (blockSize));

	FILL_1D(src, N, dis(gen));
	FILL_1D(centers, N_CLUSTERS, dis(gen));

	CUDA_CALL(cudaMemcpy, src_d, src, sizeof(float) * N, cudaMemcpyHostToDevice);
	CUDA_CALL(cudaBindTexture, NULL, tex, src_d, sizeof(float) * N);

	CUDA_CALL(cudaMemcpy, centers_d, centers, sizeof(float) * N_CLUSTERS, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate, &start);
	CUDA_CALL(cudaEventCreate, &stop);

	cudaEventRecord(start);
	for(int it = 0; it < MAX_IT; it++){
		relabel_k<<<blockSize, BLOCKDIM, N_CLUSTERS * sizeof(struct Cluster) * BLOCKDIM>>>(centers_d, labels_d);
		findCenters_k<<<1, BLOCKDIM2, N_CLUSTERS * sizeof(struct Cluster) * BLOCKDIM2>>>(centers_d);
//		for(int c = 0; c < N_CLUSTERS; c++){
//			calculateClusters_k<<<BLOCKDIM2, BLOCKDIM, sizeof(struct Cluster) * BLOCKDIM>>>(labels_d, 0);
//			findCenters_k<<<1, BLOCKDIM2, sizeof(struct Cluster) * BLOCKDIM2>>>(centers_d, c);
//		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();

	float msecs = 0.0f;
	cudaEventElapsedTime(&msecs, start, stop);

	CUDA_CALL(cudaMemcpy, labels, labels_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
	CUDA_CALL(cudaMemcpy, centers, centers_d, sizeof(float) * N_CLUSTERS, cudaMemcpyDeviceToHost);
	printf("Blocks = %d\n", blockSize);

//	PRINT_1D_I(labels, N);
	PRINT_1D_F(centers, N_CLUSTERS);
	int* freq = new int[N_CLUSTERS];
	memset(freq, 0, sizeof(int) * N_CLUSTERS);
	for(int i = 0; i < N; i++){
		freq[labels[i]]++;
	}
	int total = 0;
	for(int i = 0; i < N_CLUSTERS; i++)
		total += freq[i];
	assert(total == N);
	printf("Time = %f\n", msecs);

	CUDA_CALL(cudaEventDestroy, start);
	CUDA_CALL(cudaEventDestroy, stop);
	CUDA_CALL(cudaUnbindTexture, tex);
	CUDA_CALL(cudaFree, src_d);
//	CUDA_CALL(cudaFree, clusters_d);
	CUDA_CALL(cudaFree, centers_d);
	CUDA_CALL(cudaFree, labels_d);

	delete[] src;
	delete[] labels;
	delete[] centers;
	return 0;
}
