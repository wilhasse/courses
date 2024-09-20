#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITERATIONS 1000

void mandelbrot(unsigned char *image, float x_min, float x_max, float y_min, float y_max)
{
    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
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
}

int main()
{
    unsigned char *image = (unsigned char*)malloc(WIDTH * HEIGHT * sizeof(unsigned char));
    float x_min = -2.0f, x_max = 1.0f, y_min = -1.5f, y_max = 1.5f;
    
    clock_t start = clock();
    
    mandelbrot(image, x_min, x_max, y_min, y_max);
    
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
    
    printf("CPU execution time: %f ms\n", cpu_time_used);
    
    // Print a small portion of the result
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%3d ", image[i * WIDTH + j]);
        }
        printf("\n");
    }
    
    free(image);
    return 0;
}
