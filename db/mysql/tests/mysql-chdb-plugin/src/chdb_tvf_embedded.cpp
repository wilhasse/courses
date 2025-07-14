#include <mysql.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>

// Path to the clickhouse_data directory from mysql-to-chdb-example
const char* CLICKHOUSE_DATA_PATH = "/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data";
const char* DEBUG_LOG = "/tmp/mysql_chdb_embedded_debug.log";

// Use the deprecated but stable v2 API
struct local_result_v2 {
    char * buf;
    size_t len;
    void * _vec;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char * error_message;
};

typedef struct local_result_v2* (*query_stable_v2_fn)(int argc, char** argv);
typedef void (*free_result_v2_fn)(struct local_result_v2* result);

extern "C" {

// Helper function to log debug messages
void debug_log(const std::string& message) {
    std::ofstream log_file(DEBUG_LOG, std::ios::app);
    if (log_file.is_open()) {
        log_file << "[" << time(nullptr) << "] " << message << std::endl;
        log_file.close();
    }
}

// Global handles for the library
static void* chdb_handle = nullptr;
static query_stable_v2_fn query_stable_v2 = nullptr;
static free_result_v2_fn free_result_v2 = nullptr;
static bool lib_initialized = false;

// Initialize library
bool init_chdb() {
    if (lib_initialized) {
        debug_log("chDB already initialized");
        return true;
    }
    
    debug_log("Initializing chDB library...");
    
    const char* lib_paths[] = {
        "/home/cslog/chdb/libchdb.so",
        "/home/cslog/chdb/buildlib/libchdb.so",
        "libchdb.so"
    };
    
    for (const char* path : lib_paths) {
        debug_log("Trying to load: " + std::string(path));
        chdb_handle = dlopen(path, RTLD_LAZY);
        if (chdb_handle) {
            debug_log("Successfully loaded library from: " + std::string(path));
            break;
        } else {
            debug_log("Failed to load from " + std::string(path) + ": " + std::string(dlerror()));
        }
    }
    
    if (!chdb_handle) {
        debug_log("ERROR: Failed to load libchdb.so from any path");
        return false;
    }
    
    dlerror(); // Clear any existing error
    
    query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        debug_log("ERROR: Failed to get query_stable_v2: " + std::string(dlsym_error));
        dlclose(chdb_handle);
        chdb_handle = nullptr;
        return false;
    }
    
    free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
    dlsym_error = dlerror();
    if (dlsym_error) {
        debug_log("ERROR: Failed to get free_result_v2: " + std::string(dlsym_error));
        dlclose(chdb_handle);
        chdb_handle = nullptr;
        return false;
    }
    
    debug_log("Successfully loaded all functions");
    lib_initialized = true;
    return true;
}

// Helper function to execute chDB query
std::string execute_chdb_query(const std::string& query) {
    debug_log("execute_chdb_query called with: " + query);
    
    if (!init_chdb()) {
        debug_log("ERROR: Failed to initialize chDB");
        return "";
    }
    
    // Prepare arguments
    std::vector<char*> argv;
    std::vector<std::string> args_storage;
    
    args_storage.push_back("clickhouse");
    args_storage.push_back("--multiquery");
    args_storage.push_back("--output-format=TabSeparated");
    args_storage.push_back("--path=" + std::string(CLICKHOUSE_DATA_PATH));
    args_storage.push_back("--query=" + query);
    
    debug_log("Arguments:");
    for (size_t i = 0; i < args_storage.size(); i++) {
        argv.push_back(const_cast<char*>(args_storage[i].c_str()));
        debug_log("  argv[" + std::to_string(i) + "] = " + args_storage[i]);
    }
    
    // Execute query
    debug_log("Calling query_stable_v2...");
    struct local_result_v2* result = query_stable_v2(argv.size(), argv.data());
    
    if (!result) {
        debug_log("ERROR: query_stable_v2 returned NULL");
        return "";
    }
    
    debug_log("Result received:");
    debug_log("  buf = " + std::string(result->buf ? "non-null" : "null"));
    debug_log("  len = " + std::to_string(result->len));
    debug_log("  elapsed = " + std::to_string(result->elapsed));
    debug_log("  rows_read = " + std::to_string(result->rows_read));
    debug_log("  bytes_read = " + std::to_string(result->bytes_read));
    if (result->error_message) {
        debug_log("  error_message = " + std::string(result->error_message));
    }
    
    if (!result->buf || result->len == 0) {
        debug_log("ERROR: Empty result buffer");
        if (result->error_message) {
            debug_log("Error message: " + std::string(result->error_message));
        }
        free_result_v2(result);
        return "";
    }
    
    std::string output(result->buf, result->len);
    debug_log("Raw output: [" + output + "]");
    
    free_result_v2(result);
    
    // Remove trailing newline
    if (!output.empty() && output.back() == '\n') {
        output.pop_back();
    }
    
    debug_log("Returning: [" + output + "]");
    return output;
}

// UDF: ch_customer_count() - returns count of customers in ClickHouse
bool ch_customer_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    debug_log("ch_customer_count_init called");
    
    if (args->arg_count != 0) {
        strcpy(message, "ch_customer_count() does not accept arguments");
        return 1;
    }
    
    if (!init_chdb()) {
        strcpy(message, "Failed to initialize chDB library");
        return 1;
    }
    
    initid->maybe_null = 1;
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
        debug_log("ERROR: Empty result from count query");
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

// UDF: ch_get_customer_id(row_num) - returns customer id for given row
bool ch_get_customer_id_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_get_customer_id() requires exactly one argument");
        return 1;
    }
    
    if (!init_chdb()) {
        strcpy(message, "Failed to initialize chDB library");
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
    
    if (!init_chdb()) {
        strcpy(message, "Failed to initialize chDB library");
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
    
    if (!init_chdb()) {
        strcpy(message, "Failed to initialize chDB library");
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
    
    if (!init_chdb()) {
        strcpy(message, "Failed to initialize chDB library");
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
    
    if (!init_chdb()) {
        strcpy(message, "Failed to initialize chDB library");
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
