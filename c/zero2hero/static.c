#include <stdio.h>
#include "static.h"

// something is wrong here :(
int increaseItem() {
    static int size = 0;
    size++;
    return size;
}

static void generate_static2() {

    printf("Static function \n");
}

int generate_static() {

    generate_static2();
    printf("Static size %d\n", increaseItem());
    printf("Static size %d\n", increaseItem());
    printf("Static size %d\n", increaseItem());
    printf("Static size %d\n", increaseItem());

    return 0;
}
