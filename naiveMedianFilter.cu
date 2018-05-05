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

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 32
#define BLOCKDIM BLOCKDIM_X * BLOCKDIM_Y
#define W (512 * 200)
#define H 512
#define N W * H

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

#define KERNEL_R 3

__global__ void fill(float *a, float v) {
    size_t x = threadIdx.x;
    size_t y = blockIdx.x;
    if (x < W && y < H)
        a[x + y * W] = v;
}

__device__ void sort(float *a, float *b, float *c) {
    float d;
    if(*a > *b){
        d = *a;
        *a = *b;
        *b  = d;
    }
    if(*a > *c){
        d = *a;
        *a = *c;
        *c  = d;
    }
    if(*b > *c){
        d = *b;
        *b = *c;
        *c  = d;
    }
}

__device__ void swap(float *a, float *b){
    float d = *a;
    *a = *b;
    *b = d;
}

__device__ void insertionSort(float *src, int n){
    for(int i = 1; i < n; i++){
        int j = i - 1;
        float key = src[i];
        while(j >= 0 && src[j] > key){
            if(src[j] > src[j + 1])
                swap(&src[j], &src[j + 1]);
            j--;
        }
    }
}

__global__ void naiveMedianFilter(const float *src, int w, int h, float *dst){
    float imgBlock float[KERNEL_R * KERNEL_R];
    int r = KERNEL_R;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int rHalf = r / 2;
    int i, j, k, l;
    if(x - rHalf > 0 && x + rHalf < w && y - rHalf > 0 && y + rHalf < h){
        for (i = x - rHalf, k = 0; i <= x + rHalf; i++, k++) {
            for (j = y - rHalf, l = 0; j <= y + rHalf; j++, l++) {
                imgBlock[k + l * r] = src[i + j * w];
            }
        }

        //Sort block
        insertionSort(imgBlock, r * r);

        //Set median
        dst[x + y * w] = imgBlock[rHalf + rHalf * r];
    }
    else if(x < w && y < h){
        dst[x + y * w] = src[x + y * w];
    }
}


int main() {
    //Host variables
    float *img;
    float *res;

    //Device variables
    float *deviceImg;
    float *deviceRes;

    curandGenerator_t gen;

    img = (float *) malloc(sizeof(float) * N);
    res = (float *) malloc(sizeof(float) * N);

    CUDA_CALL(cudaMalloc, (void **) &deviceImg, sizeof(float) * N);
    CUDA_CALL(cudaMalloc, (void **) &deviceRes, sizeof(float) * N);

    CURAND_CALL(curandCreateGenerator, &gen, CURAND_RNG_PSEUDO_DEFAULT);
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed, gen, 1234ULL);

    dim3 blockSize = dim3(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 gridSize = dim3((unsigned int) ceil(W / BLOCKDIM_X), (unsigned int) ceil(H / BLOCKDIM_Y));
    CURAND_CALL(curandGenerateUniform, gen, deviceImg, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    medianFilter<<<gridSize, blockSize>>>(deviceImg, W, H, deviceRes);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CUDA_CALL(cudaMemcpy, img, deviceImg, N * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CALL(cudaMemcpy, res, deviceRes, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Source image: \n");
    PRINT_FLAT2D(img, W, H);

    printf("Blurred image: \n");
    PRINT_FLAT2D(res, W, H);

    printf("TIME = %f\n", milliseconds / 1000.0f);

    CURAND_CALL(curandDestroyGenerator, gen);
    CUDA_CALL(cudaFree, deviceImg);
    CUDA_CALL(cudaFree, deviceRes);

    CUDA_CALL(cudaEventDestroy, start);
    CUDA_CALL(cudaEventDestroy, stop);

    CUDA_CALL(cudaDeviceReset);
    free(img);
    free(res);

    return EXIT_SUCCESS;
}