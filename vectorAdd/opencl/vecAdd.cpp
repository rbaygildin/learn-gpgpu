#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


using namespace std;

template<typename T>
int* prepareData(size_t size){
    random_device device;
    mt19937 gen(device());
    uniform_int_distribution<int> dist;
    T* data = new T[size];
    for(int i = 0; i < size; i++){
        data[i] = dist(gen);
    }
    return data;
}

const char* readSrc(const char* fileName){
    ifstream in(fileName);
    stringstream stream;
    string line;
    for(;getline(in, line);){
        stream << line;
    }
    string tmp = stream.str();
    const char* arr = tmp.c_str();
    return arr;
}

const int N = 1000;

int main(int argc, char const *argv[])
{
    
    int* a = prepareData<int>(N);
    int* b = prepareData<int>(N);
    int* c = new int[N];

    cl_context ctx;
    cl_command_queue cmdQ;
    cl_program program;
    cl_kernel kernel;
    cl_context_properties props[3];
    cl_platform_id platformId;
    cl_uint nPlatforms = 0;
    cl_device_id deviceId;
    cl_uint nDevices = 0;
    cl_int err;
    if(clGetPlatformIDs(1, &platformId, &nPlatforms) != CL_SUCCESS){
        cout << "Can get platforms" << endl;
        return -1;
    }
    if(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, &nDevices) != CL_SUCCESS){
        cout << "Can get devices" << endl;
        return -1;
    }
    props[0] = CL_CONTEXT_PLATFORM;
    props[1] = (cl_context_properties) platformId;
    props[2] = 0;
    ctx = clCreateContext(props, 1, &deviceId, nullptr, nullptr, &err); 
    cmdQ = clCreateCommandQueue(ctx, deviceId, 0, &err);
    const char* prgSrc = readSrc("./kernel.cl");
    program = clCreateProgramWithSource(ctx, 1, (const char**) prgSrc, nullptr, &err);
    if(clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS){
        cout << "Can not build program" << endl;
        return -1;
    }
    kernel = clCreateKernel(program, "vecAdd", &err);
    cl_mem devA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * N, nullptr, nullptr);
    cl_mem devB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(int) * N, nullptr, nullptr);
    cl_mem devC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * N, nullptr, nullptr);

    clEnqueueWriteBuffer(cmdQ, devA, CL_TRUE, 0, sizeof(int) * N, a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cmdQ, devB, CL_TRUE, 0, sizeof(int) * N, b, 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &devA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &devB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &devC);

    size_t global;
    clEnqueueNDRangeKernel(cmdQ, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cmdQ, devC, CL_TRUE, 0, sizeof(int) * N, c, 0, NULL, NULL);

    for(int i = 0; i < N; i++){
        cout << "c[" << i << "] = a[" << i << "] + c[" << i << "] = " << c[i]; 
    }

    clReleaseMemObject(devA);
    clReleaseMemObject(devB);
    clReleaseMemObject(devC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(cmdQ);
    clReleaseContext(ctx);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
