#include <mysql.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

extern "C" {

// UDF: test2_row_count() - returns the number of rows in our virtual table
bool test2_row_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
  if (args->arg_count != 0) {
    strcpy(message, "test2_row_count() does not accept arguments");
    return 1;
  }
  initid->maybe_null = 0;
  return 0;
}

void test2_row_count_deinit(UDF_INIT *initid) {
}

long long test2_row_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
  *is_null = 0;
  *error = 0;
  return 5; // Our virtual table has 5 rows
}

// UDF: test2_get_id(row_num) - returns id for given row
bool test2_get_id_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
  if (args->arg_count != 1) {
    strcpy(message, "test2_get_id() requires exactly one argument");
    return 1;
  }
  args->arg_type[0] = INT_RESULT;
  initid->maybe_null = 1;
  return 0;
}

void test2_get_id_deinit(UDF_INIT *initid) {
}

long long test2_get_id(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
  *is_null = 0;
  *error = 0;
  
  long long row_num = *((long long*)args->args[0]);
  if (row_num < 1 || row_num > 5) {
    *is_null = 1;
    return 0;
  }
  
  return row_num; // id equals row number
}

// UDF: test2_get_name(row_num) - returns name for given row
static char name_buffer[256];

bool test2_get_name_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
  if (args->arg_count != 1) {
    strcpy(message, "test2_get_name() requires exactly one argument");
    return 1;
  }
  args->arg_type[0] = INT_RESULT;
  initid->maybe_null = 1;
  initid->max_length = 255;
  return 0;
}

void test2_get_name_deinit(UDF_INIT *initid) {
}

char *test2_get_name(UDF_INIT *initid, UDF_ARGS *args, char *result,
                     unsigned long *length, char *is_null, char *error) {
  *is_null = 0;
  *error = 0;
  
  long long row_num = *((long long*)args->args[0]);
  if (row_num < 1 || row_num > 5) {
    *is_null = 1;
    return NULL;
  }
  
  snprintf(name_buffer, sizeof(name_buffer), "Row %lld", row_num);
  *length = strlen(name_buffer);
  return name_buffer;
}

// UDF: test2_get_value(row_num) - returns value for given row
bool test2_get_value_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
  if (args->arg_count != 1) {
    strcpy(message, "test2_get_value() requires exactly one argument");
    return 1;
  }
  args->arg_type[0] = INT_RESULT;
  initid->maybe_null = 1;
  initid->decimals = 1;
  return 0;
}

void test2_get_value_deinit(UDF_INIT *initid) {
}

double test2_get_value(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
  *is_null = 0;
  *error = 0;
  
  long long row_num = *((long long*)args->args[0]);
  if (row_num < 1 || row_num > 5) {
    *is_null = 1;
    return 0.0;
  }
  
  return row_num * 10.5;
}

}