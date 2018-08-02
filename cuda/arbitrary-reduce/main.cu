#include<stdio.h>
#include<stdlib.h>

#include "helper_cuda.h"

typedef float T;

__global__ void reduce_max_kernel(const T *d_arr, int size, T *d_out)
{
    extern __shared__ T s_data[];
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int block_size = blockDim.x;

    if (blockIdx.x == gridDim.x -1)
    {
    block_size -= gridDim.x * blockDim.x - size;
    }

    if (tid < block_size) {
    s_data[tid] = d_arr[gid];
    }

    __syncthreads();

    int s = (block_size + 1) <<1;
    
    while(s>0)
    {
    if (tid < s && tid+s < block_size){
        s_data[tid] = max(s_data[tid],s_data[tid+s]);
    }

    s = min((s+1)<<1,s-1);
    __syncthreads();

    }

    if (tid == 0){
    d_out[blockIdx.x] = s_data[0];
    }

}

int reduced_max(T *h_arr, int size)
{
    T *d_arr, *d_inter, *d_out;
    T re;
    int blocks, threads_per_block;
    int inter_size;

    threads_per_block = 1024;
    blocks = (size + threads_per_block - 1) / threads_per_block;
    inter_size = blocks;
    
    checkCudaErrors(cudaMalloc((void**)&d_arr, size*sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_inter, inter_size*sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(T)));
    
    cudaMemcpy(d_arr, h_arr, size*sizeof(T),cudaMemcpyHostToDevice);
    reduce_max_kernel <<<blocks, threads_per_block, threads_per_block * sizeof(T)>>>(d_arr, size, d_inter);
    reduce_max_kernel <<<1, blocks, blocks * sizeof(T)>>>(d_inter,inter_size, d_out);

    cudaMemcpy(&re, d_out, sizeof(T), cudaMemcpyDeviceToHost);
    
    checkCudaErrors(cudaFree(d_arr));
    checkCudaErrors(cudaFree(d_inter));
    checkCudaErrors(cudaFree(d_out));
    return re;
}


int main()
{
    const int arrsize = 20000;
    T h_arr[arrsize];
    T result, expected_ans = -1;
    
    srand(time(NULL));
    
    for (int i=0; i < arrsize; i++)
    {
        h_arr[i] = rand() / 100;
        expected_ans = max(expected_ans, h_arr[i]);
    }

    printf("The minimum number: %f\n", expected_ans);

    result = reduced_max(h_arr, arrsize);

    printf("The minimum number: %f\n", result);

    return 0;
}
