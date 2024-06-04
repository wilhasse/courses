#include <stdio.h>
#include "static.h"

// something is wrong here :(
int increaseItem() {
    static int size = 0;
    size++;
    return size;
}

int generate_static() {

    printf("Static size %d\n", increaseItem());
    printf("Static size %d\n", increaseItem());
    printf("Static size %d\n", increaseItem());
    printf("Static size %d\n", increaseItem());

    return 0;
}
