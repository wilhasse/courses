#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 500000000

void vectorAdd(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    int *a, *b, *c;
    int size = N * sizeof(int);
    clock_t start, end;
    double cpu_time_used;

    // Allocate memory
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // Start timing
    start = clock();

    // Perform vector addition
    vectorAdd(a, b, c, N);

    // End timing
    end = clock();

    // Calculate time taken
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds

    printf("Time taken for CPU operation: %f ms\n", cpu_time_used);

    // Verify the result
    //for (int i = 0; i < N; i++)
    //{
    //    printf("%d + %d = %d\n", a[i], b[i], c[i]);
    //}

    // Free memory
    free(a);
    free(b);
    free(c);

    return 0;
}
