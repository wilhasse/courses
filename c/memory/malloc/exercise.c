#include <stdio.h>
#include <stdlib.h>

#include "exercise.h"

int *allocate_scalar_list(int size, int multiplier) {

  // ?
  int *c = (int*)malloc(size*sizeof(int));
  if (c == NULL) {
      // failed
      return NULL;
  }

  for (int i=0;i<size;i++) {
    c[i] = i * multiplier;
  }

  return c;
}
