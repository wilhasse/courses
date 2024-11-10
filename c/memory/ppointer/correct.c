#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 CORRECT APPROACH - Using pointer to pointer

In the correct approach:
ptr is a pointer to p
*ptr modifies the actual p in main
Changes to the pointer are visible in main

Note: allocateMemory creates a internal pointer
see correct.c to a direct approach
*/
void allocateMemory(int **ptr) {
    int *iptr = (int *)malloc(sizeof(int));
    *ptr = iptr;
    **ptr = 42;
}

int main() {
    int *p = NULL;
    allocateMemory(&p);  // p gets updated with malloc'd address
    printf("Value: %d\n", *p);  // Works correctly
    free(p);
    return 0;
}
