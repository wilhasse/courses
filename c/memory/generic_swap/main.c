#include "munit.h"
#include "bootlib.h"
#include "exercise.h"
#include <stdint.h>

typedef struct CoffeShop {
  uint64_t quality;
  uint64_t taste;
  uint64_t branding;
} coffee_shop_t;

munit_case(RUN, test_generic_ints, {
  int i1 = 1234;
  int i2 = 5678;

  swap(&i1, &i2, sizeof(int));

  munit_assert_int(i1, ==, 5678, "i1 should be i2's original value");
  munit_assert_int(i2, ==, 1234, "i2 should be i1's original value");
  munit_assert_true(boot_all_freed());
});

munit_case(RUN, test_generic_strings, {
  char *s1 = "dax";
  char *s2 = "adam";

  swap(&s1, &s2, sizeof(char *));
  munit_assert_string_equal(s1, "adam", "s1 should be s2's original value");
  munit_assert_string_equal(s2, "dax", "s2 should be s1's original value");
  munit_assert_true(boot_all_freed());
});

munit_case(SUBMIT, test_generic_structs, {
  coffee_shop_t sbucks = {2, 3, 4};
  coffee_shop_t terminalshop = {10, 10, 10};

  swap(&sbucks, &terminalshop, sizeof(coffee_shop_t));

  munit_assert_int(sbucks.quality, ==, 10, "sbucks.quality should be terminalshop.quality");
  munit_assert_int(sbucks.taste, ==, 10, "sbucks.taste should be terminalshop.taste");
  munit_assert_int(sbucks.branding, ==, 10, "sbucks.branding should be terminalshop.branding");

  munit_assert_int(terminalshop.quality, ==, 2, "terminalshop.quality should be sbucks.quality");
  munit_assert_int(terminalshop.taste, ==, 3, "terminalshop.taste should be sbucks.taste");
  munit_assert_int(terminalshop.branding, ==, 4, "terminalshop.branding should be sbucks.branding");
});

int main() {
  MunitTest tests[] = {
    munit_test("/generic_ints", test_generic_ints),
    munit_test("/generic_strings", test_generic_strings),
    munit_test("/generic_struct", test_generic_structs),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("swap", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
