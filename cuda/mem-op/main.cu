#include<stdio.h>

#define imin(a,b) (a<b?a:b)

const int N = 100 * 1024 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(int size, float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < size) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

float malloc_test(int size) {
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the CPU side
    a = (float*) malloc(size * sizeof(float));
    b = (float*) malloc(size * sizeof(float));
    partial_c = (float*) malloc(blocksPerGrid * sizeof(float));

    // allocate the memory on the GPU
    cudaMalloc((void**) &dev_a, size * sizeof(float));
    cudaMalloc((void**) &dev_b, size * sizeof(float));
    cudaMalloc((void**) &dev_partial_c, blocksPerGrid * sizeof(float));

    // fill in the host memory with data
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaEventRecord(start, 0);
    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);
    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float),
            cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    // free memory on the CPU side
    free(a);
    free(b);
    free(partial_c);

    // free events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Result:  %f\n", c);

    return elapsedTime;
}

float cuda_zero_copy_alloc_test(int size) {
    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate the memory on the CPU
    cudaHostAlloc((void**) &a, size * sizeof(float),
            cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**) &b, size * sizeof(float),
            cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**) &partial_c, blocksPerGrid * sizeof(float),
            cudaHostAllocMapped);

    // find out the GPU pointers
    cudaHostGetDevicePointer(&dev_a, a, 0);
    cudaHostGetDevicePointer(&dev_b, b, 0);
    cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0);

    // fill in the host memory with data
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    cudaEventRecord(start, 0);

    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_c[i];
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(partial_c);

    // free events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Result:  %f\n", c);

    return elapsedTime;
}

float cuda_host_alloc_test(int size) {
    cudaEvent_t start, stop;
    float *aa, *bb, c, *partial_cc;
    float *dev_aa, *dev_bb, *dev_partial_cc;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the CPU side
    cudaHostAlloc((void**) &aa, size * sizeof(*aa), cudaHostAllocDefault);
    cudaHostAlloc((void**) &bb, size * sizeof(*bb), cudaHostAllocDefault);
    cudaHostAlloc((void**) &partial_cc, size * sizeof(*partial_cc), cudaHostAllocDefault);

    // allocate the memory on the GPU
    cudaMalloc((void**) &dev_aa, size * sizeof(float));
    cudaMalloc((void**) &dev_bb, size * sizeof(float));
    cudaMalloc((void**) &dev_partial_cc, blocksPerGrid * sizeof(float));

    // fill in the host memory with data
    for (int i = 0; i < size; i++) {
        aa[i] = i;
        bb[i] = i * 2;
    }

    cudaEventRecord(start, 0);
    // copy the arrays 'a' and 'b' to the GPU

    cudaMemcpy(dev_aa, aa, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_bb, bb, size * sizeof(float), cudaMemcpyHostToDevice);


    dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_aa, dev_bb, dev_partial_cc);
    // copy the array 'c' back from the GPU to the CPU
    cudaMemcpy(partial_cc, dev_partial_cc, blocksPerGrid * sizeof(float),
            cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // finish up on the CPU side
    c = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        c += partial_cc[i];
    }

    cudaFree(dev_aa);
    cudaFree(dev_bb);
    cudaFree(dev_partial_cc);

    // free memory on the CPU side
    cudaFreeHost(aa);
    cudaFreeHost(bb);
    cudaFreeHost(partial_cc);

    // free events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Result:  %f\n", c);

    return elapsedTime;
}

int main(void) {
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (prop.canMapHostMemory != 1) {
        printf("Device can not map memory.\n");
        return 0;
    }
    float elapsedTime;

    cudaSetDeviceFlags (cudaDeviceMapHost);

    // try it with malloc
    elapsedTime = malloc_test(N);
    printf("malloc:  %3.1f ms\n", elapsedTime);

    // now try it with cudaHostAlloc
    elapsedTime = cuda_zero_copy_alloc_test(N);
    printf("zero-copy cudahostmalloc:  %3.1f ms\n", elapsedTime);

    // now try it with cudaHostAlloc
    elapsedTime = cuda_host_alloc_test(N);
    printf("cudahostmalloc:  %3.1f ms\n", elapsedTime);
}
