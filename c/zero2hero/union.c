#include <stdio.h>
#include "union.h"

int generate_union() {

    union myunion {
      int i;
      char c;
    };

    //
    union myunion my;
    my.i = 0x65;

    printf("Union 101: %c\n", my.c);
    return 0;
}
