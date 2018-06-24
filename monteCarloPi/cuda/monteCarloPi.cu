#include <cstdio>
#include <curand.h>
#include <cstdlib>
#include <random>

// #define DEBUG
// #define USE_CURAND

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

__global__ void countPoints(const int n, const float *x, const float *y, int* psum){
	extern __shared__ int bsum[];

	size_t tid = threadIdx.x;
	size_t bid = blockIdx.x;
	size_t bsize = blockDim.x;
	size_t pos = tid + bid * bsize;

	if(pos < n){
		float _x = x[pos];
		float _y = y[pos];

#ifdef DEBUG
		printf("[%d] x = %f, y = %f\n", pos, _x, _y);
#endif
		if(hypotf(_x, _y) < 1.0){
			bsum[tid] = 1;

#ifdef DEBUG
			printf("WITHIN CIRLE x = %f, y = %f\n", _x, _y);
#endif
		}
		else{
			bsum[tid] = 0;
		}
	}
	else{
		bsum[tid] = 0;
	}
	__syncthreads();

#ifdef DEBUG
	printf("%d\n", bsum[tid]);
#endif
	
	for(unsigned int stride = bsize / 2; stride > 0; stride /= 2){
		if(tid < stride){
			bsum[tid] += bsum[tid + stride];
		}
		__syncthreads();
	}
	__syncthreads();
	if(tid == 0){
		psum[bid] = bsum[0];

#ifdef DEBUG
		printf("%d\n", psum[bid]);
#endif
	}
}

#define BSIZE 1000
#define N_POINTS (BSIZE * BSIZE)

int main(int argc, char const *argv[])
{

#ifdef USE_CURAND
	curandGenerator_t gen;
#else
	std::random_device randDevice;
	std::mt19937 gen(randDevice());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
#endif

	int *psum_host = new int[BSIZE];
	float *x_host, *y_host;
	x_host = new float[N_POINTS];
	y_host = new float[N_POINTS];

	float *x_dev, *y_dev;
	bool *flags_dev;
	int *psum_dev;

	CUDA_CALL(cudaMalloc, (void**)&flags_dev, N_POINTS * sizeof(bool));
	CUDA_CALL(cudaMalloc, (void**)&psum_dev, BSIZE * sizeof(int));
	CUDA_CALL(cudaMalloc, (void**)&x_dev, N_POINTS * sizeof(float));
	CUDA_CALL(cudaMalloc, (void**)&y_dev, N_POINTS * sizeof(float));

	CUDA_CALL(cudaMemset, psum_dev, 0, BSIZE * sizeof(int));

#ifdef USE_CURAND
	CURAND_CALL(curandCreateGenerator, &gen, CURAND_RNG_PSEUDO_DEFAULT);
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed, gen, 1234ULL);
	CURAND_CALL(curandGenerateUniform, gen, x_dev, N_POINTS);
	CURAND_CALL(curandGenerateUniform, gen, y_dev, N_POINTS);
#else
	for(int i = 0; i < N_POINTS; i++){
		x_host[i] = dist(gen);
		y_host[i] = dist(gen);
	}
	CUDA_CALL(cudaMemcpy, x_dev, x_host, N_POINTS * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CALL(cudaMemcpy, y_dev, y_host, N_POINTS * sizeof(float), cudaMemcpyHostToDevice);
#endif

	countPoints<<<BSIZE, BSIZE, BSIZE * sizeof(int)>>>(N_POINTS, x_dev, y_dev, psum_dev);
	CUDA_CALL(cudaMemcpy, psum_host, psum_dev, BSIZE * sizeof(int), cudaMemcpyDeviceToHost);

	int nWithinCirclePoints = 0;
	for(int i = 0; i < BSIZE; i++){
		nWithinCirclePoints += psum_host[i];
	}

	float pi = 4.0f * nWithinCirclePoints / N_POINTS;
	printf("Num of points = %d\n", nWithinCirclePoints);
	printf("Calulated PI = %f\n", pi);

#ifdef USE_CURAND
	CURAND_CALL(curandDestroyGenerator, gen);
#endif
	CUDA_CALL(cudaFree, flags_dev);
	CUDA_CALL(cudaFree, psum_dev);
	CUDA_CALL(cudaFree, x_dev);
	CUDA_CALL(cudaFree, y_dev);
	delete[] psum_host; 
	delete[] x_host;
	delete[] y_host;
	return 0;
}