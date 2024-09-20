#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITERATIONS 10000
#define NUM_RUNS 10

__global__ void mandelbrot(unsigned char *image, double x_min, double x_max, double y_min, double y_max)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < HEIGHT && col < WIDTH) {
        double x0 = x_min + (x_max - x_min) * col / (WIDTH - 1);
        double y0 = y_min + (y_max - y_min) * row / (HEIGHT - 1);
        
        double x = 0.0;
        double y = 0.0;
        int iteration = 0;
        
        while (x*x + y*y <= 4.0 && iteration < MAX_ITERATIONS) {
            double xtemp = x*x - y*y + x0;
            y = 2*x*y + y0;
            x = xtemp;
            iteration++;
        }
        
        unsigned char color = (iteration == MAX_ITERATIONS) ? 0 : 255 * sqrtf(iteration / (double)MAX_ITERATIONS);
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
    
    // Zoomed in coordinates
    double x_min = -0.745;
    double x_max = -0.744;
    double y_min = 0.1;
    double y_max = 0.101;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0;
    
    for (int i = 0; i < NUM_RUNS; i++) {
        cudaEventRecord(start);
        
        mandelbrot<<<grid, block>>>(d_image, x_min, x_max, y_min, y_max);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        
        printf("Run %d: Kernel execution time: %f ms\n", i+1, milliseconds);
    }
    
    printf("Average kernel execution time: %f ms\n", total_time / NUM_RUNS);
    
    cudaMemcpy(h_image, d_image, WIDTH * HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Print a small portion of the result
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            printf("%3d ", h_image[i * WIDTH + j]);
        }
        printf("\n");
    }
    
    // Save image to a file (you'll need to implement this part)
    // save_image(h_image, WIDTH, HEIGHT, "mandelbrot.png");
    
    cudaFree(d_image);
    free(h_image);
    
    return 0;
}
