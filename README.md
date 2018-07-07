# Awesome GPGPU
This is a curated list of of examples of using GPU in general-purpose computings, libraries and papers.

## Examples

### CUDA

<<<<<<< HEAD
* [Vector addition](https://github.com/rbaygildin/awesome-gpgpu/tree/master/vectorAdd) - Simplest fast one-dimensional vectors addition
=======
#### Linear algebra
>>>>>>> 4286c00b8f22b19920348b593325d3013b1557f2

* [Vector addition](https://github.com/rbaygildin/awesome-gpgpu/tree/master/vectorAdd) - Simplest fast one-dimensional vectors addition [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/tree/master/vectorAdd)]

* [Sum of elements in an array](https://github.com/rbaygildin/awesome-gpgpu/blob/master/sumArray) - Parallel sum of elements in an array [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/blob/master/sumArray/sum.cu)]

#### Image processing

* [2D convolution](https://github.com/rbaygildin/awesome-gpgpu/blob/master/convolution) - Naïve implementation of 2D convolution [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/blob/master/convolution/convolve2D.cu)]

* [Median filter](https://github.com/rbaygildin/awesome-gpgpu/tree/master/medianFilter) - Median filter with arbitrary size kernel [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/tree/master/medianFilter)]

* [Sobel edge-detection filter](https://github.com/rbaygildin/awesome-gpgpu/blob/master/sobel/sobel.cu) - Parallel implementation of Sobel Operator which is used in image processing [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/blob/master/sobel/sobel.cu)] 

#### Clustering

* [K Means clustering](https://github.com/rbaygildin/awesome-gpgpu/blob/master/kmeans2/cuda_kmeans.cu) - Fast Floyd K Means on GPU. Shared memory and two-step reduction (partial and global) are used to implement finding cluster centers [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/blob/master/kmeans2/cuda_kmeans.cu)]

* [Fuzzy C Means clustering](https://github.com/rbaygildin/awesome-gpgpu/blob/master/fcm/cuda_fcm.cu) - Fuzzy C Means. Shared memory and two-step reduction (partial and global) are used to implement finding cluster centers [[CUDA](https://github.com/rbaygildin/awesome-gpgpu/blob/master/fcm/cuda_fcm.cu)]

#### Simulation

* [Calculating PI with Monte Carlo method](https://github.com/rbaygildin/awesome-gpgpu/blob/master/monteCarloPi) - Find PI with Monte Carlo method [[CPU](https://github.com/rbaygildin/awesome-gpgpu/blob/master/monteCarloPi/cpu) | [CUDA](https://github.com/rbaygildin/awesome-gpgpu/blob/master/monteCarloPi/cuda)]

* [cuBlas SAXPY](https://github.com/rbaygildin/awesome-gpgpu/blob/master/saxpy/saxpy.cu) - Implementation of SAXPY with cuBlas

## Libraries

* [CUDA](https://developer.nvidia.com/cuda-toolkit) is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).

* [Thrust](https://thrust.github.io/) is a powerful library of parallel algorithms and data structures. Thrust provides a flexible, high-level interface for GPU programming that greatly enhances developer productivity. Using Thrust, C++ developers can write just a few lines of code to perform GPU-accelerated sort, scan, transform, and reduction operations orders of magnitude faster than the latest multi-core CPUs. For example, the thrust::sort algorithm delivers 5x to 100x faster sorting performance than STL and TBB.

* [OpenCL](https://www.khronos.org/opencl/) is the open, royalty-free standard for cross-platform, parallel programming of diverse processors found in personal computers, servers, mobile devices and embedded platforms. OpenCL greatly improves the speed and responsiveness of a wide spectrum of applications in numerous market categories including gaming and entertainment titles, scientific and medical software, professional creative tools, vision processing, and neural network training and inferencing.

* [Boost.Compute](http://boostorg.github.io/compute/) is a GPU/parallel-computing library for C++ based on OpenCL. The core library is a thin C++ wrapper over the OpenCL API and provides access to compute devices, contexts, command queues and memory buffers. On top of the core library is a generic, STL-like interface providing common algorithms (e.g. transform(), accumulate(), sort()) along with common containers (e.g. vector<T>, flat_set<T>). It also features a number of extensions including parallel-computing algorithms (e.g. exclusive_scan(), scatter(), reduce()) and a number of fancy iterators (e.g. transform_iterator<>, permutation_iterator<>, zip_iterator<>).

* [PyCUDA](https://documen.tician.de/pycuda/) lets you access Nvidia‘s CUDA parallel computation API from Python. Several wrappers of the CUDA API already exist–so what’s so special about PyCUDA?

* [PyOpenCL](https://documen.tician.de/pyopencl/) gives you easy, Pythonic access to the OpenCL parallel computation API. 

* [OpenACC](https://www.openacc.org/) is a user-driven directive-based performance-portable parallel programming model designed for scientists and engineers interested in porting their codes to a wide-variety of heterogeneous HPC hardware platforms and architectures with significantly less programming effort than required with a low-level model.

* [Hemi](http://harrism.github.io/hemi/) simplifies writing portable CUDA C/C++ code. With Hemi, you can write parallel kernels like you write for loops in line in your CPU code and run them on your GPUю

* [CUDPP](https://github.com/cudpp/cudpp) is the CUDA Data Parallel Primitives Library. CUDPP is a library of data-parallel algorithm primitives such as parallel-prefix-sum ("scan"), parallel sort and parallel reduction. Primitives such as these are important building blocks for a wide variety of data-parallel algorithms, including sorting, stream compaction, and building data structures such as trees and summed-area tables.


## Other awesome lists and repositories

* [Awesome CUDA by Erkaman](https://github.com/Erkaman/Awesome-CUDA) is a list of useful libraries and resources for CUDA development

* [CUDA Awesome by gmarciani](https://github.com/gmarciani/cudawesome) is a collection of awesome algorithms, implemented in CUDA
