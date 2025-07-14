#include <mysql.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

// Path to the clickhouse_data directory from mysql-to-chdb-example
const char* CLICKHOUSE_DATA_PATH = "/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data";
const char* CHDB_BINARY = "/home/cslog/chdb/buildlib/programs/clickhouse-local";
const char* DEBUG_LOG = "/tmp/mysql_chdb_debug.log";

extern "C" {

// Helper function to log debug messages
void debug_log(const std::string& message) {
    std::ofstream log_file(DEBUG_LOG, std::ios::app);
    if (log_file.is_open()) {
        log_file << "[" << time(nullptr) << "] " << message << std::endl;
        log_file.close();
    }
}

// Helper function to execute chDB query
std::string execute_chdb_query(const std::string& query) {
    std::string cmd = std::string(CHDB_BINARY) + 
                     " --path='" + CLICKHOUSE_DATA_PATH + 
                     "' --query=\"" + query + "\" --format=TabSeparated 2>&1";
    
    debug_log("Executing command: " + cmd);
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        debug_log("ERROR: Failed to open pipe");
        return "";
    }
    
    std::string result;
    std::string all_output;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        all_output += buffer;
        // Only add to result if it doesn't look like an error
        if (strstr(buffer, "Error") == nullptr && 
            strstr(buffer, "WARNING") == nullptr &&
            strstr(buffer, "Exception") == nullptr) {
            result += buffer;
        }
    }
    
    int exit_code = pclose(pipe);
    debug_log("Command output: " + all_output);
    debug_log("Exit code: " + std::to_string(WEXITSTATUS(exit_code)));
    
    // Remove trailing newline
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    debug_log("Returning result: " + result);
    return result;
}

// UDF: ch_customer_count() - returns count of customers in ClickHouse
bool ch_customer_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    debug_log("ch_customer_count_init called");
    if (args->arg_count != 0) {
        strcpy(message, "ch_customer_count() does not accept arguments");
        return 1;
    }
    initid->maybe_null = 1;  // Allow NULL returns for debugging
    return 0;
}

void ch_customer_count_deinit(UDF_INIT *initid) {
    debug_log("ch_customer_count_deinit called");
}

long long ch_customer_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    debug_log("ch_customer_count called");
    *is_null = 0;
    *error = 0;
    
    std::string result = execute_chdb_query("SELECT COUNT(*) FROM mysql_import.customers");
    if (result.empty()) {
        debug_log("ERROR: Empty result from query");
        *is_null = 1;
        return 0;
    }
    
    try {
        long long count = std::stoll(result);
        debug_log("Parsed count: " + std::to_string(count));
        return count;
    } catch (const std::exception& e) {
        debug_log("ERROR: Failed to parse result: " + std::string(e.what()));
        *is_null = 1;
        return 0;
    }
}

// UDF: ch_query_scalar(query) - execute any query returning a single value
static char scalar_buffer[1024];

bool ch_query_scalar_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    debug_log("ch_query_scalar_init called");
    if (args->arg_count != 1) {
        strcpy(message, "ch_query_scalar() requires exactly one argument");
        return 1;
    }
    args->arg_type[0] = STRING_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 1023;
    return 0;
}

void ch_query_scalar_deinit(UDF_INIT *initid) {
    debug_log("ch_query_scalar_deinit called");
}

char *ch_query_scalar(UDF_INIT *initid, UDF_ARGS *args, char *result,
                     unsigned long *length, char *is_null, char *error) {
    debug_log("ch_query_scalar called");
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        debug_log("ERROR: NULL query argument");
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    debug_log("Query: " + query);
    
    std::string res = execute_chdb_query(query);
    
    if (res.empty()) {
        debug_log("ERROR: Empty result");
        *is_null = 1;
        return NULL;
    }
    
    strncpy(scalar_buffer, res.c_str(), sizeof(scalar_buffer) - 1);
    scalar_buffer[sizeof(scalar_buffer) - 1] = '\0';
    *length = strlen(scalar_buffer);
    debug_log("Returning: " + std::string(scalar_buffer));
    return scalar_buffer;
}

} // extern "C"
