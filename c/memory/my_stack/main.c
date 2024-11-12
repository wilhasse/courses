#include "exercise.h"

//#include "bootlib.h"
#include "munit.h"
#include "snekstack.h"

munit_case(RUN, multiple_types_stack, {
  stack_t *s = stack_new(4);
  munit_assert_not_null(s, "Must allocate a new stack");

  stack_push_multiple_types(s);
  munit_assert_int(s->count, ==, 2, "Should have two items in the stack");

  float *f = s->data[0];
  munit_assert_float_equal(*f, 3.14, "Float is equal");

  char *string = s->data[1];
  munit_assert_string_equal(string, "Sneklang is blazingly slow!", "char* is equal");

  free(f);
  free(string);
  stack_free(s);
});

int main() {
  MunitTest tests[] = {
    munit_test("/multiple_types_stack", multiple_types_stack),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("snekstack", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
