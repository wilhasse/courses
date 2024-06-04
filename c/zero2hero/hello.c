#include <stdio.h>
#include "string.h"
#include "union.h"
#include "struct.h"
#include "pointer.h"
#include "static.h"
#include "leak.h"

int main() {

  int my_ids[32];
  my_ids[3] = 2;
  my_ids[0] = 2;

  printf("Hello, World!\n");
  printf("%d\n", my_ids[3]);
  generate_string();
  generate_union();
  generate_struct();
  generate_pointer();
  generate_static();
  generate_leak();

  return 0;
}
