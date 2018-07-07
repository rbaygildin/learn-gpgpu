__kernel void vecAdd(__global int* a, __global int* b, __global int* c){
    size_t pos = get_global_id(0);
    c[pos] = a[pos] + b[pos];
}