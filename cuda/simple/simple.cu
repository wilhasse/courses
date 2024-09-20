#include "simple.h"

__device__ float deviceMultiply(float a, float b)
{
    return a * b;
}

__global__ void vectorMult(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = deviceMultiply(A[i], B[i]);
    }
}

__host__ std::tuple<float *, float *, float *> allocateHostMemory(int numElements)
{
    size_t size = numElements * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    return {h_A, h_B, h_C};
}

__host__ std::tuple<float *, float *, float *> allocateDeviceMemory(int numElements)
{
    float *d_A, *d_B, *d_C;
    size_t size = numElements * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    return {d_A, d_B, d_C};
}

__host__ void copyFromHostToDevice(float *h_A, float *h_B, float *d_A, float *d_B, int numElements)
{
    size_t size = numElements * sizeof(float);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
}

__host__ void executeKernel(float *d_A, float *d_B, float *d_C, int numElements)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
}

__host__ void copyFromDeviceToHost(float *d_C, float *h_C, int numElements)
{
    size_t size = numElements * sizeof(float);
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void deallocateMemory(float *h_A, float *h_B, float *h_C, float *d_A, float *d_B, float *d_C)
{
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

__host__ void cleanUpDevice()
{
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void performTest(float *h_A, float *h_B, float *h_C, int numElements)
{
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs((h_A[i] * h_B[i]) - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");
}

int main(void)
{
    int numElements = 100000000;
    printf("[Vector multiplication of %d elements]\n", numElements);

    auto [h_A, h_B, h_C] = allocateHostMemory(numElements);
    auto [d_A, d_B, d_C] = allocateDeviceMemory(numElements);
    copyFromHostToDevice(h_A, h_B, d_A, d_B, numElements);

    executeKernel(d_A, d_B, d_C, numElements);

    copyFromDeviceToHost(d_C, h_C, numElements);
    performTest(h_A, h_B, h_C, numElements);
    deallocateMemory(h_A, h_B, h_C, d_A, d_B, d_C);

    cleanUpDevice();
    printf("Done\n");
    return 0;
}
