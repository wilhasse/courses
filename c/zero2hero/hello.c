#include <stdio.h>
#include "string.h"
#include "union.h"

int main() {

  int my_ids[32];
  my_ids[3] = 2;
  my_ids[0] = 2;

  printf("Hello, World!\n");
  printf("%d\n", my_ids[3]);
  generate_string();
  generate_union();
  generate_struct();

  return 0;
}
