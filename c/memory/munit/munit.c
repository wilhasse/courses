/*
 Generated by Claude Sonnet
*/
#include "munit.h"

// Global verbose mode flag
int munit_verbose_mode = 0;

// Add this to your main function or test runner
void enable_verbose_mode() {
    munit_verbose_mode = 1;
}

MunitSuite munit_suite(const char* name, MunitTest* tests) {
    MunitSuite suite = {
        .name = name,
        .tests = tests
    };
    return suite;
}

MunitTest* munit_suite_get_tests(MunitSuite* suite) {
    return suite->tests;
}

static void run_test(MunitTest* test) {
    if (test->name && test->test) {
        printf("Running test %s...", test->name);
        test->test();
        printf(" OK\n");
    }
}

int munit_suite_main(MunitSuite* suite, void* user_data, int argc, char* const* argv) {
    (void)user_data;  // Unused parameters
    (void)argc;
    (void)argv;

    // Enable verbose mode if desired
    enable_verbose_mode();

    printf("\nRunning test suite: %s\n", suite->name);
    printf("----------------------------------------\n");

    MunitTest* current_test = suite->tests;
    int test_count = 0;

    while (current_test->name != NULL && current_test->test != NULL) {
        run_test(current_test);
        current_test++;
        test_count++;
    }

    printf("----------------------------------------\n");
    printf("All %d tests passed!\n\n", test_count);

    return 0;
}

