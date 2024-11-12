#include <stdio.h>
#include <stdlib.h>

#include "munit.h"
#include "snekobject.h"

munit_case(RUN, test_array_set, {
  snek_object_t *foo = new_snek_integer(1);

  snek_object_t *array = new_snek_array(1);
  snek_array_set(array, 0, foo);
  munit_assert_int(foo->refcount, ==, 2, "foo is now referenced by array");

  //munit_assert(!boot_is_freed(foo));

  refcount_dec(foo);
  refcount_dec(array);
  //munit_assert(boot_all_freed());
});

munit_case(SUBMIT, test_array_free, {
  snek_object_t *foo = new_snek_integer(1);
  snek_object_t *bar = new_snek_integer(2);
  snek_object_t *baz = new_snek_integer(3);

  snek_object_t *array = new_snek_array(2);
  snek_array_set(array, 0, foo);
  snek_array_set(array, 1, bar);
  munit_assert_int(foo->refcount, ==, 2, "foo is now referenced by array");
  munit_assert_int(bar->refcount, ==, 2, "bar is now referenced by array");
  munit_assert_int(baz->refcount, ==, 1, "baz is not yet referenced by array");

  // `foo` is stil referenced in the `array`, so it should not be freed.
  refcount_dec(foo);
  //munit_assert(!boot_is_freed(foo));

  // Overwrite index 0, which is `foo`, with `baz`.
  //  Now `foo` is not referenced by `array`, so it should be freed.
  snek_array_set(array, 0, baz);
  //munit_assert(boot_is_freed(foo));

  refcount_dec(bar);
  refcount_dec(baz);
  refcount_dec(array);
  //munit_assert(boot_all_freed());
});

munit_case(RUN, test_vector3_refcounting, {
  snek_object_t *foo = new_snek_integer(1);
  snek_object_t *bar = new_snek_integer(2);
  snek_object_t *baz = new_snek_integer(3);

  snek_object_t *vec = new_snek_vector3(foo, bar, baz);
  munit_assert_int(foo->refcount, ==, 2, "foo is now referenced by vec");
  munit_assert_int(bar->refcount, ==, 2, "bar is now referenced by vec");
  munit_assert_int(baz->refcount, ==, 2, "baz is now referenced by vec");

  // `foo` is stil referenced in the `vec`, so it should not be freed.
  refcount_dec(foo);
  //munit_assert(!boot_is_freed(foo));

  refcount_dec(vec);
  //munit_assert(boot_is_freed(foo));

  // These are still alive, they have the original reference still.
  //munit_assert(!boot_is_freed(bar));
  //munit_assert(!boot_is_freed(baz));

  // Decrement the last reference to the objects, so they will be freed.
  refcount_dec(bar);
  refcount_dec(baz);

  //munit_assert(boot_all_freed());
});

munit_case(RUN, test_int_has_refcount, {
  snek_object_t *obj = new_snek_integer(10);
  munit_assert_int(obj->refcount, ==, 1, "Refcount should be 1 on creation");

  free(obj);
});

munit_case(RUN, test_inc_refcount, {
  snek_object_t *obj = new_snek_float(4.20);
  munit_assert_int(obj->refcount, ==, 1, "Refcount should be 1 on creation");

  refcount_inc(obj);
  munit_assert_int(obj->refcount, ==, 2, "Refcount should be incremented");

  free(obj);
});

munit_case(RUN, test_dec_refcount, {
  snek_object_t *obj = new_snek_float(4.20);

  refcount_inc(obj);
  munit_assert_int(obj->refcount, ==, 2, "Refcount should be incremented");

  refcount_dec(obj);
  munit_assert_int(obj->refcount, ==, 1, "Refcount should be decremented");

  //munit_assert(!boot_is_freed(obj));

  // Object is still alive, so we will free manually.
  free(obj);
});

munit_case(RUN, test_refcount_free_is_called, {
  snek_object_t *obj = new_snek_float(4.20);

  refcount_inc(obj);
  munit_assert_int(obj->refcount, ==, 2, "Refcount should be incremented");

  refcount_dec(obj);
  munit_assert_int(obj->refcount, ==, 1, "Refcount should be decremented");

  refcount_dec(obj);
  //munit_assert(boot_is_freed(obj));
  //munit_assert(boot_all_freed());
});

munit_case(RUN, test_allocated_string_is_freed, {
  snek_object_t *obj = new_snek_string("Hello @wagslane!");

  refcount_inc(obj);
  munit_assert_int(obj->refcount, ==, 2, "Refcount should be incremented");

  refcount_dec(obj);
  munit_assert_int(obj->refcount, ==, 1, "Refcount should be decremented");
  munit_assert_string_equal(obj->data.v_string, "Hello @wagslane!", "references str");

  refcount_dec(obj);
  //munit_assert(boot_is_freed(obj));
  //munit_assert(boot_all_freed());
});

int main() {
  MunitTest tests[] = {
    munit_test("/array_set", test_array_set),
    munit_test("/array_free", test_array_free),
    munit_test("/has_refcount", test_int_has_refcount),
    munit_test("/inc_refcount", test_inc_refcount),
    munit_test("/dec_refcount", test_dec_refcount),
    munit_test("/free_refcount", test_refcount_free_is_called),
    munit_test("/string_freed", test_allocated_string_is_freed),
    munit_test("/vector3", test_vector3_refcounting),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("refcount", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
