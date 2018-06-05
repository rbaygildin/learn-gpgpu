#include <assert.h>
#include <cstdio>
#include <random>

using namespace std;

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
};

__device__ float euclidian_dist(const float a, const float b){
	return (a - b) * (a - b);
}

__global__ void relabel_k(const float* src, const float* clusters, int n, int nClusters, int* labels){
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if(pos < n){
		float minDist = 1.0f;
		int clusterIndex = 0;
		for(int c = 0; c < nClusters; c++){
			float dist = euclidian_dist(src[pos], clusters[c]);
			if(dist <= minDist){
				clusterIndex = c;
				minDist = dist;
			}
		}
		labels[pos] = clusterIndex;
	}
}

__global__ void calculateClusters_k(const float* src, const int* labels, int n, int clusterIndex, Cluster* dst){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	_clusters[tid] = Cluster();
	_clusters[tid].sum = 0.0f;
	_clusters[tid].count = 0;
	if(pos < n && labels[pos] == clusterIndex){
		_clusters[tid].sum = src[pos];
		_clusters[tid].count = 1;
	}
	__syncthreads();
	for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
		if(threadIdx.x < stride){
			_clusters[tid].sum += _clusters[tid + stride].sum;
			_clusters[tid].count += _clusters[tid + stride].count;
		}
		__syncthreads();
	}
	__syncthreads();
	if(threadIdx.x == 0){
		dst[blockIdx.x].sum = _clusters[0].sum;
		dst[blockIdx.x].count = _clusters[0].count;
		printf("BlockId = %d, Sum = %f, Count = %d\n", blockIdx.x, _clusters[0].sum, _clusters[0].count);
	}
}

__global__ void findCenters_k(const Cluster* src, int n, int clusterIndex, float* dst){
	extern __shared__ Cluster _clusters[];
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	_clusters[tid] = src[pos];
	__syncthreads();
	for(unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2){
		if(threadIdx.x < stride){
			_clusters[tid].sum += _clusters[tid + stride].sum;
			_clusters[tid].count += _clusters[tid + stride].count;
		}
		__syncthreads();
	}
	__syncthreads();
	if(tid == 0){
		printf("Cluster = %d, Sum = %f, Count = %d\n", clusterIndex, _clusters[0].sum, _clusters[0].count);
		dst[clusterIndex] = _clusters[0].count > 0 ? _clusters[0].sum / _clusters[0].count : 0.0f;
	}
}

int main(){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0.0, 1.0);

	const int N = 512 * 512;
	const int N_CLUSTERS = 5;
	const int BLOCKDIM = 1024;
	const int MAX_IT = 10;
	const int blockSize = (N + BLOCKDIM - 1) / BLOCKDIM;
	float* src = new float[N];
	int* labels = new int[N];
	float* centers = new float[N_CLUSTERS];

	float* src_d, *centers_d;
	Cluster* clusters_d;
	int* labels_d;

	CUDA_CALL(cudaMalloc, (void**)&src_d, sizeof(float) * N);
	CUDA_CALL(cudaMalloc, (void**)&labels_d, sizeof(int) * N);
	CUDA_CALL(cudaMalloc, (void**)&centers_d, sizeof(float) * N_CLUSTERS);
	CUDA_CALL(cudaMalloc, (void**)&clusters_d, sizeof(struct Cluster) * (blockSize));

	FILL_1D(src, N, dis(gen));
	FILL_1D(centers, N_CLUSTERS, dis(gen));

	CUDA_CALL(cudaMemcpy, src_d, src, sizeof(float) * N, cudaMemcpyHostToDevice);
	CUDA_CALL(cudaMemcpy, centers_d, centers, sizeof(float) * N_CLUSTERS, cudaMemcpyHostToDevice);

	for(int it = 0; it < MAX_IT; it++){
		relabel_k<<<blockSize, BLOCKDIM>>>(src_d, centers_d, N, N_CLUSTERS, labels_d);
		for(int c = 0; c < N_CLUSTERS; c++){
			calculateClusters_k<<<blockSize, BLOCKDIM, sizeof(struct Cluster) * BLOCKDIM>>>(src_d, labels_d, N, c, clusters_d);
			findCenters_k<<<1, blockSize, sizeof(struct Cluster) * blockSize>>>(clusters_d, N, c, centers_d);
		}
	}
	cudaDeviceSynchronize();

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
	CUDA_CALL(cudaFree, src_d);
	CUDA_CALL(cudaFree, clusters_d);
	CUDA_CALL(cudaFree, centers_d);
	CUDA_CALL(cudaFree, labels_d);

	delete[] src;
	delete[] labels;
	delete[] centers;
	return EXIT_SUCCESS;
}
