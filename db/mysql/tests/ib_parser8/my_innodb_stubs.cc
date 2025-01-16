// my_innodb_stubs.cc
#include <cstdio>
#include <cstdlib>

// MySQL 5.7 includes
#include "univ.i"

extern "C" int WriteCoreDump() {
   return 1; // do nothing
}
