#include <mysql.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>

// Path to the clickhouse_data directory from mysql-to-chdb-example
const char* CLICKHOUSE_DATA_PATH = "/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data";
const char* LIBCHDB_PATH = "/home/cslog/chdb/libchdb.so";

// chDB structures
struct local_result_v2 {
    char * buf;
    size_t len;
    void * _vec; // std::vector<char> *, for freeing
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
};

extern "C" {

// Global handles for the library
static void* chdb_handle = nullptr;
static struct local_result_v2* (*query_stable_v2)(int, char**) = nullptr;
static void (*free_result_v2)(struct local_result_v2*) = nullptr;

// Initialize library
bool init_chdb() {
    if (chdb_handle) return true;
    
    chdb_handle = dlopen(LIBCHDB_PATH, RTLD_LAZY);
    if (!chdb_handle) {
        return false;
    }
    
    *(void**)(&query_stable_v2) = dlsym(chdb_handle, "query_stable_v2");
    if (!query_stable_v2) {
        dlclose(chdb_handle);
        chdb_handle = nullptr;
        return false;
    }
    
    *(void**)(&free_result_v2) = dlsym(chdb_handle, "free_result_v2");
    if (!free_result_v2) {
        dlclose(chdb_handle);
        chdb_handle = nullptr;
        return false;
    }
    
    return true;
}

// Helper function to execute chDB query
std::string execute_chdb_query(const std::string& query) {
    if (!init_chdb()) {
        return "";
    }
    
    // Prepare arguments
    std::vector<char*> argv;
    std::vector<std::string> arg_strings;
    
    arg_strings.push_back("clickhouse");
    arg_strings.push_back("--multiquery");
    arg_strings.push_back("--output-format=TabSeparated");
    arg_strings.push_back("--path=" + std::string(CLICKHOUSE_DATA_PATH));
    arg_strings.push_back("--query=" + query);
    
    for (auto& arg : arg_strings) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }
    
    // Execute query
    struct local_result_v2* result = query_stable_v2(argv.size(), argv.data());
    if (!result || !result->buf) {
        if (result) free_result_v2(result);
        return "";
    }
    
    std::string output(result->buf, result->len);
    free_result_v2(result);
    
    // Remove trailing newline
    if (!output.empty() && output.back() == '\n') {
        output.pop_back();
    }
    
    return output;
}

// UDF: ch_customer_count() - returns count of customers in ClickHouse
bool ch_customer_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 0) {
        strcpy(message, "ch_customer_count() does not accept arguments");
        return 1;
    }
    initid->maybe_null = 1;
    return 0;
}

void ch_customer_count_deinit(UDF_INIT *initid) {}

long long ch_customer_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    std::string result = execute_chdb_query("SELECT COUNT(*) FROM mysql_import.customers");
    if (result.empty()) {
        *is_null = 1;
        return 0;
    }
    
    try {
        return std::stoll(result);
    } catch (...) {
        *is_null = 1;
        return 0;
    }
}

// UDF: ch_get_customer_id(row_num) - returns customer id for given row
bool ch_get_customer_id_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_get_customer_id() requires exactly one argument");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    return 0;
}

void ch_get_customer_id_deinit(UDF_INIT *initid) {}

long long ch_get_customer_id(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return 0;
    }
    
    std::stringstream query;
    query << "SELECT id FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result = execute_chdb_query(query.str());
    if (result.empty()) {
        *is_null = 1;
        return 0;
    }
    
    try {
        return std::stoll(result);
    } catch (...) {
        *is_null = 1;
        return 0;
    }
}

// UDF: ch_get_customer_name(row_num) - returns customer name for given row
static char name_buffer[256];

bool ch_get_customer_name_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_get_customer_name() requires exactly one argument");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 255;
    return 0;
}

void ch_get_customer_name_deinit(UDF_INIT *initid) {}

char *ch_get_customer_name(UDF_INIT *initid, UDF_ARGS *args, char *result,
                          unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return NULL;
    }
    
    std::stringstream query;
    query << "SELECT name FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string res = execute_chdb_query(query.str());
    if (res.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(name_buffer, res.c_str(), sizeof(name_buffer) - 1);
    name_buffer[sizeof(name_buffer) - 1] = '\0';
    *length = strlen(name_buffer);
    return name_buffer;
}

// UDF: ch_get_customer_city(row_num) - returns customer city for given row
static char city_buffer[256];

bool ch_get_customer_city_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_get_customer_city() requires exactly one argument");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 255;
    return 0;
}

void ch_get_customer_city_deinit(UDF_INIT *initid) {}

char *ch_get_customer_city(UDF_INIT *initid, UDF_ARGS *args, char *result,
                          unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return NULL;
    }
    
    std::stringstream query;
    query << "SELECT city FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string res = execute_chdb_query(query.str());
    if (res.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(city_buffer, res.c_str(), sizeof(city_buffer) - 1);
    city_buffer[sizeof(city_buffer) - 1] = '\0';
    *length = strlen(city_buffer);
    return city_buffer;
}

// UDF: ch_get_customer_age(row_num) - returns customer age for given row
bool ch_get_customer_age_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_get_customer_age() requires exactly one argument");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    return 0;
}

void ch_get_customer_age_deinit(UDF_INIT *initid) {}

long long ch_get_customer_age(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return 0;
    }
    
    std::stringstream query;
    query << "SELECT age FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result = execute_chdb_query(query.str());
    if (result.empty()) {
        *is_null = 1;
        return 0;
    }
    
    try {
        return std::stoll(result);
    } catch (...) {
        *is_null = 1;
        return 0;
    }
}

// UDF: ch_query_scalar(query) - execute any query returning a single value
static char scalar_buffer[1024];

bool ch_query_scalar_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_query_scalar() requires exactly one argument");
        return 1;
    }
    args->arg_type[0] = STRING_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 1023;
    return 0;
}

void ch_query_scalar_deinit(UDF_INIT *initid) {}

char *ch_query_scalar(UDF_INIT *initid, UDF_ARGS *args, char *result,
                     unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    std::string res = execute_chdb_query(query);
    
    if (res.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(scalar_buffer, res.c_str(), sizeof(scalar_buffer) - 1);
    scalar_buffer[sizeof(scalar_buffer) - 1] = '\0';
    *length = strlen(scalar_buffer);
    return scalar_buffer;
}

} // extern "C"
