#include<stdio.h>
#include<sys/time.h>
#include "helper_cuda.h"

void cpu_hist(unsigned char *h, int size, int *result){
    for(int i=0;i<size;i++)
    result[h[i]] += 1;
    return; 
}

__global__ void hist_kernel(unsigned char *d, int size, unsigned int *d_result){

__shared__ unsigned int block_shm[256];

block_shm[threadIdx.x] = 0;
__syncthreads();

int tid = threadIdx.x + blockIdx.x*blockDim.x;
int thread_num = blockDim.x*gridDim.x;

while(tid <size)
{
atomicAdd(&block_shm[int(d[tid])],1);
tid+=thread_num;
}

__syncthreads();

atomicAdd(&d_result[threadIdx.x],block_shm[threadIdx.x]);

}

void gpu_hist(unsigned char *h, int size, unsigned int *result){
    unsigned char *d;
    unsigned int *d_result;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
    checkCudaErrors(cudaMalloc((void **)&d, size*sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void **)&d_result, 256*sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(d,h,size*sizeof(unsigned char),cudaMemcpyHostToDevice));
    
    unsigned int grid = 2 * prop.multiProcessorCount;
    unsigned int block = 256;
    unsigned int shm = 256*sizeof(unsigned int); 
    hist_kernel<<< grid, block, shm>>>(d, size, d_result);
    checkCudaErrors(cudaMemcpy(result, d_result, 256*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    cudaFree(d);
    cudaFree(d_result);
}

int main(){
    int SIZE = 100<<20;
    unsigned int hist[256];
    int expected_hist[256];
    unsigned char *h;
    h = (unsigned char*) malloc(SIZE*sizeof(unsigned char));

    for(int i=0;i<SIZE;i++)
    h[i] = rand()&0xff;

    for(int i=0;i<256;i++)
    expected_hist[i] = 0;
    
    struct timeval start_time, stop_time;
    double elapsed_time;

    gettimeofday(&start_time, NULL);
    cpu_hist(h, SIZE, expected_hist);
    gettimeofday(&stop_time, NULL);
    elapsed_time = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;

    printf("CPU : %f ms\n", elapsed_time);


    gettimeofday(&start_time, NULL);
    gpu_hist(h, SIZE, hist);
    gettimeofday(&stop_time, NULL);
    elapsed_time = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;

    printf("GPU : %f ms\n", elapsed_time);

    bool flag = true;

    for(int i=0;i<256;i++)
    if(expected_hist[i]!=hist[i]) flag = false;

    if(flag)
    printf("Correct!\n");

}
