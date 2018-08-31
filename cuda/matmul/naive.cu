#include<stdlib.h>
#include "helper_cuda.h"

__device__ float get_element(float* Mat, int row, int col, int row_size)
{ return Mat[row*row_size+col]; }

__device__ void set_element(float* Mat, float &value, int row, int col, int row_size)
{ Mat[row*row_size+col] = value; }

__global__ void matmul(float *A, float *B, float *C, int row_size_a, int row_size_b)
{

int tid_col = threadIdx.x + blockDim.x*blockIdx.x;
int tid_row = threadIdx.y + blockDim.y*blockIdx.y;

float tmp = 0.0;

for(int i = 0 ; i<row_size_a; i++)
{

tmp += get_element(A, tid_row, i, row_size_a) * get_element( B, i, tid_col, row_size_b );

}

set_element(C, tmp, tid_row, tid_col, row_size_b);

}

int main(){

float *A, *B, *C;

int rows = 8192;
int cols = 8192;
int cc = 8192;

A = (float*) malloc(rows*cols*sizeof(float));
B = (float*) malloc(cols*cc*sizeof(float));
C = (float*) malloc(rows*cc*sizeof(float));

float *d_A, *d_B, *d_C;
checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(float)*rows*cols));
checkCudaErrors(cudaMalloc((void **)&d_B, sizeof(float)*cc*cols));
checkCudaErrors(cudaMalloc((void **)&d_C, sizeof(float)*rows*cc));

checkCudaErrors(cudaMemcpy(d_A, A, sizeof(float)*rows*cols, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_B, B, sizeof(float)*cc*cols, cudaMemcpyHostToDevice));
checkCudaErrors(cudaMemcpy(d_C, C, sizeof(float)*rows*cc, cudaMemcpyHostToDevice));

dim3 blocks(32,32);
dim3 grids((rows+31)/32,(cc+31)/32);

matmul<<< grids, blocks>>>(d_A, d_B, d_C, cols, cc);

checkCudaErrors(cudaMemcpy(C, d_C, sizeof(float)*rows*cc, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(d_A));
checkCudaErrors(cudaFree(d_B));
checkCudaErrors(cudaFree(d_C));

return 0;
}
