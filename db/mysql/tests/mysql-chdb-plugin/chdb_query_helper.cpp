#include <iostream>
#include <string>
#include <dlfcn.h>
#include <cstring>
#include <vector>

// This helper program loads chDB and executes queries
// It runs as a separate process to avoid crashing MySQL

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

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: chdb_query_helper 'SQL_QUERY'" << std::endl;
        return 1;
    }
    
    std::string query = argv[1];
    
    // Load libchdb.so
    void* chdb_handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
    if (!chdb_handle) {
        std::cerr << "Failed to load libchdb.so: " << dlerror() << std::endl;
        return 1;
    }
    
    // Get function pointers
    query_stable_v2_fn query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
    free_result_v2_fn free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
    
    if (!query_stable_v2 || !free_result_v2) {
        std::cerr << "Failed to load functions: " << dlerror() << std::endl;
        dlclose(chdb_handle);
        return 1;
    }
    
    // Prepare arguments
    std::vector<char*> argv_query;
    std::vector<std::string> args_storage;
    
    args_storage.push_back("clickhouse");
    args_storage.push_back("--multiquery");
    args_storage.push_back("--output-format=TabSeparated");
    args_storage.push_back("--path=/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data");
    args_storage.push_back("--query=" + query);
    
    for (auto& arg : args_storage) {
        argv_query.push_back(const_cast<char*>(arg.c_str()));
    }
    
    // Execute query
    struct local_result_v2* result = query_stable_v2(argv_query.size(), argv_query.data());
    
    if (!result) {
        dlclose(chdb_handle);
        return 1;
    }
    
    // Output result
    if (result->buf && result->len > 0) {
        std::cout.write(result->buf, result->len);
    }
    
    // Cleanup
    free_result_v2(result);
    dlclose(chdb_handle);
    
    return 0;
}
