#include <stdlib.h>

#include "munit.h"
#include "snekobject.h"

munit_case(RUN, test_integer_add, {
  snek_object_t *one = new_snek_integer(1);
  snek_object_t *three = new_snek_integer(3);
  snek_object_t *four = snek_add(one, three);

  munit_assert_not_null(four, "must return an object");
  munit_assert_int(four->kind, ==, INTEGER, "1 + 3 = 4");
  munit_assert_int(four->data.v_int, ==, 4, "1 + 3 = 4");

  free(one);
  free(three);
  free(four);
  munit_assert_true(boot_all_freed());
});

munit_case(RUN, test_float_add, {
  snek_object_t *one = new_snek_float(1.5);
  snek_object_t *three = new_snek_float(3.5);
  snek_object_t *five = snek_add(one, three);

  munit_assert_not_null(five, "must return an object");
  munit_assert_int(five->kind, ==, FLOAT, "1.5 + 3.5 = 5.0");
  munit_assert_float(five->data.v_float, ==, 1.5 + 3.5, "1.5 + 3.5 = 5.0");

  free(one);
  free(three);
  free(five);
  munit_assert_true(boot_all_freed());
});

munit_case(RUN, test_string_add, {
  snek_object_t *hello = new_snek_string("hello");
  snek_object_t *world = new_snek_string(", world");
  snek_object_t *greeting = snek_add(hello, world);

  munit_assert_not_null(greeting, "must return an object");
  munit_assert_int(greeting->kind, ==, STRING, "Must be a string!");
  munit_assert_string_equal(
      greeting->data.v_string, "hello, world", "Should concatenate strings"
  );

  free(hello->data.v_string);
  free(hello);
  free(world->data.v_string);
  free(world);
  free(greeting->data.v_string);
  free(greeting);
  munit_assert_true(boot_all_freed());
});

munit_case(SUBMIT, test_string_add_self, {
  snek_object_t *repeated = new_snek_string("(repeated)");
  snek_object_t *result = snek_add(repeated, repeated);

  munit_assert_not_null(result, "must return an object");
  munit_assert_int(result->kind, ==, STRING, "Must be a string!");
  munit_assert_string_equal(
    result->data.v_string,
    "(repeated)(repeated)",
    "Should concatenate strings"
  );

  free(repeated->data.v_string);
  free(repeated);
  free(result->data.v_string);
  free(result);
  munit_assert_true(boot_all_freed());
});

munit_case(SUBMIT, test_vector3_add, {
  snek_object_t *one = new_snek_float(1.0);
  snek_object_t *two = new_snek_float(2.0);
  snek_object_t *three = new_snek_float(3.0);
  snek_object_t *four = new_snek_float(4.0);
  snek_object_t *five = new_snek_float(5.0);
  snek_object_t *six = new_snek_float(6.0);

  snek_object_t *v1 = new_snek_vector3(one, two, three);
  snek_object_t *v2 = new_snek_vector3(four, five, six);
  snek_object_t *result = snek_add(v1, v2);

  munit_assert_not_null(result, "must return an object");
  munit_assert_int(result->kind, ==, VECTOR3, "Must be a vector3");

  munit_assert_float(result->data.v_vector3.x->data.v_float, ==, 5.0, "x component should be 5.0");
  munit_assert_float(result->data.v_vector3.y->data.v_float, ==, 7.0, "y component should be 7.0");
  munit_assert_float(result->data.v_vector3.z->data.v_float, ==, 9.0, "z component should be 9.0");


  free(v1->data.v_vector3.x);
  free(v1->data.v_vector3.y);
  free(v1->data.v_vector3.z);
  free(v1);

  free(v2->data.v_vector3.x);
  free(v2->data.v_vector3.y);
  free(v2->data.v_vector3.z);
  free(v2);

  free(result->data.v_vector3.x);
  free(result->data.v_vector3.y);
  free(result->data.v_vector3.z);
  free(result);
  munit_assert_true(boot_all_freed());
});

munit_case(SUBMIT, test_array_add, {
    snek_object_t *one = new_snek_integer(1);
    snek_object_t *ones = new_snek_array(2);
    munit_assert(snek_array_set(ones, 0, one), "Failed to set first element in ones array");
    munit_assert(snek_array_set(ones, 1, one), "Failed to set second element in ones array");

    snek_object_t *hi = new_snek_string("hi");
    snek_object_t *hellos = new_snek_array(3);
    munit_assert(snek_array_set(hellos, 0, hi), "Failed to set first element in hellos array");
    munit_assert(snek_array_set(hellos, 1, hi), "Failed to set second element in hellos array");
    munit_assert(snek_array_set(hellos, 2, hi), "Failed to set third element in hellos array");

    snek_object_t *result = snek_add(ones, hellos);
    munit_assert(result != NULL, "must return an object");
    munit_assert(result->kind == ARRAY, "Must be an array");

    snek_object_t *first = snek_array_get(result, 0);
    munit_assert(first != NULL, "should find the first item");
    munit_assert(first->data.v_int == 1, "First item should be an int with 1");

    snek_object_t *third = snek_array_get(result, 2);
    munit_assert(third != NULL, "should find the third item");
    munit_assert(strcmp(third->data.v_string, "hi") == 0, "third == hi");

    free(one);
    free(ones->data.v_array.elements);
    free(ones);
    free(hi->data.v_string);
    free(hi);
    free(hellos->data.v_array.elements);
    free(hellos);
    free(result->data.v_array.elements);
    free(result);
    munit_assert_true(boot_all_freed());
})

int main() {
  MunitTest tests[] = {
    munit_test("/integer", test_integer_add),
    munit_test("/float", test_float_add),
    munit_test("/string", test_string_add),
    munit_test("/string-repeated", test_string_add_self),
    munit_test("/array", test_array_add),
    munit_test("/vector3", test_vector3_add),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("object-add", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
