#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <dlfcn.h>
#include <cstring>
#include <chrono>
#include "common.h"

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

int main() {
    void* chdb_handle = dlopen("libchdb.so", RTLD_LAZY);
    if (!chdb_handle) {
        std::cerr << "Failed to load libchdb.so: " << dlerror() << std::endl;
        return 1;
    }
    
    auto query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
    auto free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
    
    if (!query_stable_v2 || !free_result_v2) {
        std::cerr << "Failed to load functions: " << dlerror() << std::endl;
        return 1;
    }
    
    std::cout << "Testing chdb inserts..." << std::endl;
    
    // Test 1: Create database
    {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=CREATE DATABASE IF NOT EXISTS test_db");
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "Creating database..." << std::endl;
        auto result = query_stable_v2(argv.size(), argv.data());
        if (result) {
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            } else {
                std::cout << "Success" << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    // Test 2: Create MergeTree table
    {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=CREATE TABLE IF NOT EXISTS test_db.test_merge (id Int32, value String) ENGINE = MergeTree() ORDER BY id");
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "\nCreating MergeTree table..." << std::endl;
        auto result = query_stable_v2(argv.size(), argv.data());
        if (result) {
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            } else {
                std::cout << "Success" << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    // Test 3: Small insert
    {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=INSERT INTO test_db.test_merge VALUES (1, 'test1'), (2, 'test2'), (3, 'test3')");
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "\nInserting 3 rows..." << std::endl;
        auto start = std::chrono::steady_clock::now();
        auto result = query_stable_v2(argv.size(), argv.data());
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (result) {
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            } else {
                std::cout << "Success (took " << duration << " ms)" << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    // Test 4: Large batch insert
    {
        std::stringstream query;
        query << "INSERT INTO test_db.test_merge VALUES ";
        for (int i = 0; i < 1000; i++) {
            if (i > 0) query << ", ";
            query << "(" << i << ", 'value" << i << "')";
        }
        
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query.str());
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "\nInserting 1000 rows in one batch..." << std::endl;
        auto start = std::chrono::steady_clock::now();
        auto result = query_stable_v2(argv.size(), argv.data());
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (result) {
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            } else {
                std::cout << "Success (took " << duration << " ms)" << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    // Test 5: Multiple sequential inserts
    std::cout << "\nTesting 10 sequential inserts of 100 rows each..." << std::endl;
    for (int batch = 0; batch < 10; batch++) {
        std::stringstream query;
        query << "INSERT INTO test_db.test_merge VALUES ";
        for (int i = 0; i < 100; i++) {
            if (i > 0) query << ", ";
            int id = batch * 100 + i + 10000;
            query << "(" << id << ", 'batch" << batch << "_value" << i << "')";
        }
        
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query.str());
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        auto start = std::chrono::steady_clock::now();
        auto result = query_stable_v2(argv.size(), argv.data());
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::cout << "  Batch " << batch << ": ";
        if (result) {
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            } else {
                std::cout << "Success (took " << duration << " ms)" << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    // Verify count
    {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=SELECT COUNT(*) FROM test_db.test_merge");
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "\nVerifying count..." << std::endl;
        auto result = query_stable_v2(argv.size(), argv.data());
        if (result) {
            if (result->buf) {
                std::cout << "Total rows: " << result->buf << std::endl;
            }
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    dlclose(chdb_handle);
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}