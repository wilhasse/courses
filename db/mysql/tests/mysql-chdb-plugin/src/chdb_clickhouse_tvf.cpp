#include <mysql.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>

// Path to the clickhouse_data directory from mysql-to-chdb-example
const char* CLICKHOUSE_DATA_PATH = "/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data";
const char* CHDB_BINARY = "/home/cslog/chdb/buildlib/programs/chl";

extern "C" {

// Helper function to execute chDB query
std::string execute_chdb_query(const std::string& query) {
    std::string cmd = std::string(CHDB_BINARY) + 
                     " --path='" + CLICKHOUSE_DATA_PATH + 
                     "' --query=\"" + query + "\" --format=TabSeparated 2>/dev/null";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    
    std::string result;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);
    
    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    return result;
}

// UDF: ch_customer_count() - returns count of customers in ClickHouse
bool ch_customer_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 0) {
        strcpy(message, "ch_customer_count() does not accept arguments");
        return 1;
    }
    initid->maybe_null = 0;
    return 0;
}

void ch_customer_count_deinit(UDF_INIT *initid) {}

long long ch_customer_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    std::string result = execute_chdb_query("SELECT COUNT(*) FROM mysql_import.customers");
    if (result.empty()) {
        *error = 1;
        return 0;
    }
    
    return std::stoll(result);
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
    
    // Query to get id from specific row (0-indexed in LIMIT)
    std::stringstream query;
    query << "SELECT id FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result = execute_chdb_query(query.str());
    if (result.empty()) {
        *is_null = 1;
        return 0;
    }
    
    return std::stoll(result);
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
    
    return std::stoll(result);
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