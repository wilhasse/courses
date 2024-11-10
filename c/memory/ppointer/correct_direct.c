#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/*
 CORRECT APPROACH - Using pointer to pointer

 It doesn't use internal variable in allocateMemory
 See correct.c
*/
void allocateMemory(int **ptr) {
    *ptr = (int *)malloc(sizeof(int));  // Modifies the original pointer
    **ptr = 42;
}

int main() {
    int *p = NULL;
    allocateMemory(&p);  // p gets updated with malloc'd address
    printf("Value: %d\n", *p);  // Works correctly
    free(p);
    return 0;
}
