#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* WRONG APPROACH - Won't work

if we used a single pointer, any changes to the pointer inside the function 
wouldn't be reflected in the main function

ptr is a copy of p
When we modify ptr, we're only changing the local copy
The original p in main remains NULL

*/
void allocateMemory(int *ptr) {
    ptr = (int *)malloc(sizeof(int));  // This change is local to the function
    *ptr = 42;
}

int main() {
    int *p = NULL;
    allocateMemory(p);  // p is still NULL after this call
    printf("Value: %d\n", *p);  // CRASH! Dereferencing NULL pointer
    return 0;
}
