#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Alternative approach - returning the pointer

This approach has several advantages:

Cleaner syntax - no need for double pointers
More intuitive - follows standard function return patterns
Similar to how many standard C functions work (like malloc, strdup, etc.)

It doesn't work if you need to create two pointers inside allocateMemory
because you can only return one element in C, in this case need to uses pointer to pointer
*/
int* allocateMemory() {
    int* ptr = (int*)malloc(sizeof(int));
    *ptr = 42;
    return ptr;
}

int main() {
    int *p = allocateMemory();  // Directly assign returned pointer
    printf("Value: %d\n", *p);  // Prints: 42
    free(p);
    return 0;
}
