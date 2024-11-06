#include "munit.h"
#include "coord.h"

munit_case(RUN, test_new_coordinate, {
  coordinate_t c = new_coord(1, 2, 3);

  assert_int(c.x, ==, 1, "should set x");
  assert_int(c.y, ==, 2, "should set y");
  assert_int(c.z, ==, 3, "should set z");
});

munit_case(RUN, test_scale_coordinate, {
  coordinate_t c = new_coord(1, 2, 3);
  coordinate_t scaled = scale_coordinate(c, 2);

  assert_int(scaled.x, ==, 2, "should scale x");
  assert_int(scaled.y, ==, 4, "should scale y");
  assert_int(scaled.z, ==, 6, "should scale z");
});

int main() {
  MunitTest tests[] = {
    munit_test("/create_coordinate", test_new_coordinate),
    munit_test("/test_scale_coordinate", test_scale_coordinate),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("coordinates", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
