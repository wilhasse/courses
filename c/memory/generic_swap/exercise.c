#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// New definition of malloc to track memory leak
#include "bootlib.h"

void swap(void *vp1, void *vp2, size_t size) {
  // ?
  void *c = (void*)malloc(size);
  if (c == NULL) {
      // failed
      return;
  }
  memcpy(c,vp2,size);
  memcpy(vp2,vp1,size);
  memcpy(vp1,c,size);
  free(c);
}
