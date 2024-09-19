#ifndef SIMPLE_H
#define SIMPLE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tuple>

// Function declarations
__global__ void vectorMult(const float *A, const float *B, float *C, int numElements);
__host__ std::tuple<float *, float *, float *> allocateHostMemory(int numElements);
__host__ std::tuple<float *, float *, float *> allocateDeviceMemory(int numElements);
__host__ void copyFromHostToDevice(float *h_A, float *h_B, float *d_A, float *d_B, int numElements);
__host__ void executeKernel(float *d_A, float *d_B, float *d_C, int numElements);
__host__ void copyFromDeviceToHost(float *d_C, float *h_C, int numElements);
__host__ void deallocateMemory(float *h_A, float *h_B, float *h_C, float *d_A, float *d_B, float *d_C);
__host__ void cleanUpDevice();
__host__ void performTest(float *h_A, float *h_B, float *h_C, int numElements);

#endif // SIMPLE_H
