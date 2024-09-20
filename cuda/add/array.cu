#include <stdio.h>
#include <cuda_runtime.h>

#define N 500000000
#define THREADS_PER_BLOCK 1

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}

int main(void)
{
    int *h_a, *h_b, *h_c;    // Host vectors
    int *d_a, *d_b, *d_c;    // Device vectors
    int size = N * sizeof(int);
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Allocate host memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; i++)
    {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Create CUDA events for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time taken for GPU operation: %f ms\n", milliseconds);

    // Verify the result
    //for (int i = 0; i < N; i++)
    //{
    //    printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    //}

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
