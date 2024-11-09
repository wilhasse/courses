#include <stdio.h>

/*
 Here c points to the same address, it gets rewritten for each new struct
*/
typedef struct {
  int x;
  int y;
} coord_t;

coord_t new_coord(int x, int y) {
  coord_t c;
  c.x = x;
  c.y = y;
  return c;
}

int main() {
  coord_t c;
  coord_t *c1;
  coord_t *c2;
  coord_t *c3;

  c = new_coord(10, 20);
  c1 = &c;
  c = new_coord(30, 40);
  c2 = &c;
  c = new_coord(50, 60);
  c3 = &c;

  printf("c1: %d, %d\n", c1->x, c1->y);
  printf("c2: %d, %d\n", c2->x, c2->y);
  printf("c3: %d, %d\n", c3->x, c3->y);
}
