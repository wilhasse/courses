#include <stdio.h>
#include <stdlib.h>
#include "leak.h"

int generate_leak() {

    int *my_int = malloc(sizeof(int));
    *my_int = 3;
    printf("Memory leaked my_int %d \n",*my_int);
    return 0;
}
