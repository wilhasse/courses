#include "munit.h"
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include "exercise.h"

// Helper function to check if a pointer is on the stack
bool is_on_stack(void *ptr) {
  void *stack_top = __builtin_frame_address(0);
  uintptr_t stack_top_addr = (uintptr_t)stack_top;
  uintptr_t ptr_addr = (uintptr_t)ptr;

  // Check within a threshold in both directions (e.g., 1MB)
  uintptr_t threshold = 1024;

  return ptr_addr >= (stack_top_addr - threshold) && ptr_addr <= (stack_top_addr + threshold);
}

munit_case(RUN, test_basic_greeting, {
  char *result = get_full_greeting("Hello", "Alice", 20);
  munit_assert_string_equal(result, "Hello Alice", "Basic greeting should be correct");
  munit_assert_false(is_on_stack(result),"Result should not be on stack");
  free(result);
});

munit_case(SUBMIT, test_short_buffer, {
  char *result = get_full_greeting("Hey", "Bob", 4);
  munit_assert_string_equal(result, "Hey", "Should truncate to fit buffer");
  munit_assert_false(is_on_stack(result),"Result should not be on stack");
  free(result);
});


int main() {
  MunitTest tests[] = {
    munit_test("/test_basic_greeting", test_basic_greeting),
    munit_test("/test_short_buffer", test_short_buffer),
    munit_null_test,
  };

  MunitSuite suite = munit_suite("get_full_greeting", tests);

  return munit_suite_main(&suite, NULL, 0, NULL);
}
