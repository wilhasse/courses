#include <stdio.h>
#include "pointer.h"

// something is wrong here :(
void swap( int *a, int *b) {
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

int generate_pointer() {

    int xx = 3;
    int *pxx = &xx;

    printf("Pointer x %d Address %p\n", *pxx, pxx);

    int x = 10, y = 20;
    printf("%d %d\n", x, y);
    swap(&x, &y);
    printf("%d %d\n", x, y);

    return 0;
}
