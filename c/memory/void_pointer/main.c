#include "munit.h"
#include "exercise.h"

munit_case(RUN, test_zero_out_integer, {
  snek_int_t integer;
  integer.value = 42;
  snek_zero_out(&integer, INTEGER);
  munit_assert_int(integer.value, ==, 0, "Integer should be zeroed out to 0");
});

munit_case(RUN, test_zero_out_float, {
  snek_float_t float_num;
  float_num.value = 3.14;
  snek_zero_out(&float_num, FLOAT);
  munit_assert_float(float_num.value, ==, 0.0, "Float should be zeroed out to 0.0");
});

munit_case(SUBMIT, test_zero_out_bool, {
  snek_bool_t boolean;
  boolean.value = 1;
  snek_zero_out(&boolean, BOOL);
  munit_assert_int(boolean.value, ==, 0, "Boolean should be zeroed out to 0");
});

munit_case(SUBMIT, test_zero_out_nonzero_values, {
  snek_int_t integer;
  snek_float_t float_num;
  snek_bool_t boolean;

  integer.value = -100;
  float_num.value = -99.99;
  boolean.value = 255;

  snek_zero_out(&integer, INTEGER);
  snek_zero_out(&float_num, FLOAT);
  snek_zero_out(&boolean, BOOL);

  munit_assert_int(integer.value, ==, 0, "Negative integer should be zeroed out to 0");
  munit_assert_float(float_num.value, ==, 0.0, "Negative float should be zeroed out to 0.0");
  munit_assert_int(boolean.value, ==, 0, "Non-zero boolean should be zeroed out to 0");
});

int main() {
    MunitTest tests[] = {
      munit_test("/test_zero_out_integer", test_zero_out_integer),
      munit_test("/test_zero_out_float", test_zero_out_float),
      munit_test("/test_zero_out_bool", test_zero_out_bool),
      munit_test("/test_zero_out_nonzero_values", test_zero_out_nonzero_values),
      munit_null_test,
    };

    MunitSuite suite = munit_suite("snek_zero_out", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}
