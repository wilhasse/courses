#include <mysql.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <sstream>
#include <unistd.h>
#include <sys/wait.h>

// This version uses a separate process to query chDB, avoiding MySQL crashes

extern "C" {

// Helper function to execute chDB query via separate process
std::string execute_chdb_query(const std::string& query) {
    // Create the command to run our query helper
    std::string cmd = "/home/cslog/courses/db/mysql/tests/mysql-chdb-plugin/chdb_query_helper ";
    cmd += "\"" + query + "\"";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }
    
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
