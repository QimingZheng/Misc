#include<stdlib.h>
#include<stdio.h>
#include "helper_cuda.h"

__global__ void assign(int *d, int SIZE){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int i=0;
    int DIM = blockDim.x*gridDim.x;
    while(tid + i*DIM <SIZE)
    {
        d[tid+i*DIM] = tid+i*DIM;
        i++;
    }
}

int main(){
    int SIZE = 10240;
    int *h = new int[SIZE];
    int *d;
    checkCudaErrors(cudaMalloc((void**)&d, SIZE * sizeof(int))); 
    checkCudaErrors(cudaMemcpy(d, h, SIZE * sizeof(int), cudaMemcpyHostToDevice));
    assign<<<64,64>>>(d,SIZE);
    checkCudaErrors(cudaMemcpy(h, d, SIZE * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d)); 
    bool flag = true;
    for (int i=0; i< SIZE; i++)
    if(h[i] != i) {printf("Error!"); flag = false; break;}
    if(flag) printf("Success!");
}
