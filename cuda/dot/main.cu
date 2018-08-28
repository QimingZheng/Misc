#include<stdio.h>
#include"helper_cuda.h"

const int SIZE = 35 * 1024;
const int ThreadPerBlock = 256;
const int BlockPerGrid = min( 32, (SIZE + ThreadPerBlock -1)/ThreadPerBlock);

__global__ void dot(float *a, float *b, float *p_c)
{
    __shared__ float cache[ThreadPerBlock];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    float tmp = 0.0;
    while(tid < SIZE){
        tmp += a[tid]*b[tid];
        tid += blockDim.x*gridDim.x;
    }

    cache[threadIdx.x] = tmp;

    __syncthreads();

    int i = blockDim.x/2;
    while(i){
        if(threadIdx.x < i){
            cache[threadIdx.x] += cache[threadIdx.x+i];
        }
        i >>= 1;
        __syncthreads();
    }
    if (threadIdx.x==0){
        p_c[blockIdx.x] = cache[0];
    }

}

int main(){
    float *h_a, *h_b, h_c, *h_partial_c;
    float *d_a, *d_b, *d_partial_c;
    h_a = (float *)malloc(SIZE*sizeof(float));
    h_b = (float *)malloc(SIZE*sizeof(float));
    h_partial_c = (float*)malloc(BlockPerGrid*sizeof(float));

    checkCudaErrors(cudaMalloc((void**)&d_a, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_b, SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_partial_c, BlockPerGrid * sizeof(float)));

    for(int i=0;i<SIZE;i++)
    {
    h_a[i] = i;
    h_b[i] = 3*i;
    }

    checkCudaErrors(cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice)); 
    checkCudaErrors(cudaMemcpy(d_partial_c, h_partial_c, BlockPerGrid * sizeof(float), cudaMemcpyHostToDevice)); 

    dot<<<BlockPerGrid, ThreadPerBlock>>>(d_a,d_b,d_partial_c);

    checkCudaErrors(cudaMemcpy(h_partial_c, d_partial_c, BlockPerGrid * sizeof(float), cudaMemcpyDeviceToHost)); 
    h_c = 0.0;
    for(int i=0;i<BlockPerGrid;i++)
    h_c += h_partial_c[i];
    
    printf("Sum = %.6f \n", h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_c);

    free(h_a);
    free(h_b);
    free(h_partial_c);

    return 0;
}
