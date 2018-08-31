#include<stdlib.h>
#include "helper_cuda.h"

#define TILE 32

__global__ void matmul(int m, int n, int k, float *A, float *B, float *C)
{
    __shared__ float ds_A[TILE][TILE]; 
    __shared__ float ds_B[TILE][TILE];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE + ty;
    int Col = bx * TILE + tx;

    float Cvalue = 0;

    for (int t=0; t<(n-1)/TILE+1; ++t)
    {
        if (Row < m && t * TILE + tx < n)
            ds_A[tx][ty] = A[Row*n+t*TILE+tx];
        else 
            ds_A[tx][ty] = 0.0;

        if (t * TILE + ty < n && Col < k)
            ds_B[tx][ty] = B[(t*TILE + ty)*k+Col];
        else
            ds_B[tx][ty] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            Cvalue += ds_A[i][ty] * ds_B[tx][i];

        __syncthreads();

        if(Row < m && Col < k)
            C[Row*k+Col]=Cvalue;
    }
}

int main(){

float *A, *B, *C;

int m = 8192;
int n = 8192;
int k = 8192;

A = (float*) malloc(m*n*sizeof(float));
B = (float*) malloc(n*k*sizeof(float));
C = (float*) malloc(m*k*sizeof(float));

float *d_A, *d_B, *d_C;
checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(float)*m*n));
checkCudaErrors(cudaMalloc((void **)&d_B, sizeof(float)*k*n));
checkCudaErrors(cudaMalloc((void **)&d_C, sizeof(float)*m*k));

checkCudaErrors(cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_B, B, sizeof(float)*k*n, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_C, C, sizeof(float)*m*k, cudaMemcpyHostToDevice));

dim3 blocks(TILE,TILE);
dim3 grids((k+TILE-1)/TILE,(m+TILE-1)/TILE);

matmul<<< grids, blocks>>>(m, n, k, d_A, d_B, d_C);

checkCudaErrors(cudaMemcpy(C, d_C, sizeof(float)*m*k, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(d_A));
checkCudaErrors(cudaFree(d_B));
checkCudaErrors(cudaFree(d_C));

return 0;
}
