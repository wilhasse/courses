// my_innodb_stubs.cc
#include <cstdio>
#include <cstdlib>

// MySQL 5.7 includes
#include "univ.i"

// If code references these:
void ut_dbg_assertion_failed(const char* expr, const char* file, unsigned long line)
{
  fprintf(stderr, "Assertion failed: %s at %s:%lu\n", expr, file, line);
  // or do nothing
}
