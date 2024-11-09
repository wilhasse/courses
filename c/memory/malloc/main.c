#include "munit.h"
#include "exercise.h"

munit_case(RUN, test_allocate_scalar_list_size, {
  int size = 5;
  int multiplier = 2;
  int *result = allocate_scalar_list(size, multiplier);
  munit_assert_not_null(result, "Function should return a non-null pointer");
  free(result);
});

munit_case(RUN, test_allocate_scalar_list_values, {
  int size = 5;
  int multiplier = 2;
  int *result = allocate_scalar_list(size, multiplier);
  int expected[5];
  expected[0] = 0;
  expected[1] = 2;
  expected[2] = 4;
  expected[3] = 6;
  expected[4] = 8;
  for (int i = 0; i < size; i++) {
    munit_assert_int(result[i], ==, expected[i], "Element does not match expected value");
  }
  free(result);
});

munit_case(SUBMIT, test_allocate_scalar_list_zero_multiplier, {
  int size = 3;
  int multiplier = 0;
  int *result = allocate_scalar_list(size, multiplier);
  for (int i = 0; i < size; i++) {
    munit_assert_int(result[i], ==, 0, "All elements should be 0 with multiplier 0");
  }
  free(result);
});

munit_case(SUBMIT, test_allocate_too_much, {
  int size = 1024 * 1024 * 100;
  int multiplier = 1;
  int *result = allocate_scalar_list(size, multiplier);
  munit_assert_null(result, "Giant allocation should result in NULL");
});

int main() {
  MunitTest tests[] = {
    munit_test("/test_allocate_scalar_list_size", test_allocate_scalar_list_size),
    munit_test("/test_allocate_scalar_list_values", test_allocate_scalar_list_values),
    munit_test("/test_allocate_scalar_list_zero_multiplier", test_allocate_scalar_list_zero_multiplier),
    munit_test("/test_allocate_too_much", test_allocate_too_much),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("allocate_scalar_list", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
