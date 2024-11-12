#include "snekstack.h"
#include "stdlib.h"
#include <string.h>

void stack_push_multiple_types(stack_t *s) {
  // ?
  float *f = (float*)malloc(sizeof(float));
  *f = 3.14;
  stack_push(s,f);

  char *c = (char*)malloc(27*sizeof(char));
  strcat (c,"Sneklang is blazingly slow!");
  stack_push(s,c);
}
