#include <iostream>
#include <dlfcn.h>
#include <cstring>

// Function pointers for chdb
typedef struct chdb_connection_* (*chdb_connect_fn)(int argc, char** argv);
typedef void (*chdb_close_conn_fn)(struct chdb_connection_* conn);
typedef struct chdb_result_* (*chdb_query_fn)(struct chdb_connection_* conn, const char* query, const char* format);
typedef void (*chdb_destroy_query_result_fn)(struct chdb_result_* result);
typedef char* (*chdb_result_buffer_fn)(struct chdb_result_* result);
typedef const char* (*chdb_result_error_fn)(struct chdb_result_* result);

// Opaque types
struct chdb_connection_ { void* internal_data; };
struct chdb_result_ { void* internal_data; };

int main() {
    // Load library
    void* handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }
    
    // Load functions
    auto chdb_connect = (chdb_connect_fn)dlsym(handle, "chdb_connect");
    auto chdb_query = (chdb_query_fn)dlsym(handle, "chdb_query");
    auto chdb_destroy_query_result = (chdb_destroy_query_result_fn)dlsym(handle, "chdb_destroy_query_result");
    auto chdb_result_buffer = (chdb_result_buffer_fn)dlsym(handle, "chdb_result_buffer");
    auto chdb_result_error = (chdb_result_error_fn)dlsym(handle, "chdb_result_error");
    auto chdb_close_conn = (chdb_close_conn_fn)dlsym(handle, "chdb_close_conn");
    
    if (!chdb_connect || !chdb_query) {
        std::cerr << "Failed to load functions" << std::endl;
        return 1;
    }
    
    std::cout << "Testing different connection methods..." << std::endl;
    
    // Test 1: Simple connection
    std::cout << "\nTest 1: Simple connection with path" << std::endl;
    char* argv1[] = {
        (char*)"clickhouse",
        (char*)"--path=./test_db",
        nullptr
    };
    auto conn1 = chdb_connect(2, argv1);
    std::cout << "Result: " << (conn1 ? "SUCCESS" : "FAILED") << std::endl;
    if (conn1) {
        // Test query
        auto result = chdb_query(conn1, "SELECT 1", "CSV");
        if (result) {
            auto error = chdb_result_error(result);
            if (error && strlen(error) > 0) {
                std::cout << "Query error: " << error << std::endl;
            } else {
                auto buffer = chdb_result_buffer(result);
                std::cout << "Query result: " << (buffer ? buffer : "NULL") << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        chdb_close_conn(conn1);
    }
    
    // Test 2: Memory connection
    std::cout << "\nTest 2: Memory connection" << std::endl;
    char* argv2[] = {
        (char*)"clickhouse",
        nullptr
    };
    auto conn2 = chdb_connect(1, argv2);
    std::cout << "Result: " << (conn2 ? "SUCCESS" : "FAILED") << std::endl;
    if (conn2) {
        // Test query
        auto result = chdb_query(conn2, "SELECT 2", "CSV");
        if (result) {
            auto error = chdb_result_error(result);
            if (error && strlen(error) > 0) {
                std::cout << "Query error: " << error << std::endl;
            } else {
                auto buffer = chdb_result_buffer(result);
                std::cout << "Query result: " << (buffer ? buffer : "NULL") << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        chdb_close_conn(conn2);
    }
    
    // Test 3: Just path
    std::cout << "\nTest 3: Just path argument" << std::endl;
    char* argv3[] = {
        (char*)"./test_db2",
        nullptr
    };
    auto conn3 = chdb_connect(1, argv3);
    std::cout << "Result: " << (conn3 ? "SUCCESS" : "FAILED") << std::endl;
    if (conn3) {
        chdb_close_conn(conn3);
    }
    
    dlclose(handle);
    return 0;
}