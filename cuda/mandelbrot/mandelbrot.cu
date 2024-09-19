#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITERATIONS 1000

__global__ void mandelbrot(unsigned char *image, float x_min, float x_max, float y_min, float y_max)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < HEIGHT && col < WIDTH) {
        float x0 = x_min + (x_max - x_min) * col / (WIDTH - 1);
        float y0 = y_min + (y_max - y_min) * row / (HEIGHT - 1);

        float x = 0.0f;
        float y = 0.0f;
        int iteration = 0;

        while (x*x + y*y <= 4.0f && iteration < MAX_ITERATIONS) {
            float xtemp = x*x - y*y + x0;
            y = 2*x*y + y0;
            x = xtemp;
            iteration++;
        }

        unsigned char color = (iteration == MAX_ITERATIONS) ? 0 : 255 * sqrtf(iteration / (float)MAX_ITERATIONS);
        image[row * WIDTH + col] = color;
    }
}

int main()
{
    unsigned char *h_image = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    unsigned char *d_image;
    
    cudaMalloc(&d_image, WIDTH * HEIGHT * sizeof(unsigned char));
    
    dim3 block(32, 32);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    
    float x_min = -2.0f, x_max = 1.0f, y_min = -1.5f, y_max = 1.5f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    mandelbrot<<<grid, block>>>(d_image, x_min, x_max, y_min, y_max);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Here you would typically save the image to a file
    // For simplicity, we'll just print a small portion of the result
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%3d ", h_image[i * WIDTH + j]);
        }
        printf("\n");
    }
    
    cudaFree(d_image);
    free(h_image);
    
    return 0;
}
