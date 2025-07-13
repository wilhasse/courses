#include <iostream>
#include <string>
#include <dlfcn.h>
#include <cstring>
#include "common.h"

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

class ClickHouseQuerier {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    ClickHouseQuerier() : chdb_handle(nullptr) {}
    
    ~ClickHouseQuerier() {
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool loadChdbLibrary() {
        const char* lib_paths[] = {
            "/home/cslog/chdb/libchdb.so",
            "./libchdb.so",
            "libchdb.so"
        };
        
        for (const char* path : lib_paths) {
            chdb_handle = dlopen(path, RTLD_LAZY);
            if (chdb_handle) {
                std::cout << "Loaded chdb library from: " << path << std::endl;
                break;
            }
        }
        
        if (!chdb_handle) {
            std::cerr << "Failed to load libchdb.so: " << dlerror() << std::endl;
            return false;
        }
        
        query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
        free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
        
        if (!query_stable_v2 || !free_result_v2) {
            std::cerr << "Failed to load functions: " << dlerror() << std::endl;
            return false;
        }
        
        return true;
    }
    
    struct local_result_v2* executeQuery(const std::string& query, const std::string& output_format = "CSV") {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--output-format=" + output_format);
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        return query_stable_v2(argv.size(), argv.data());
    }
    
    void verifyData() {
        std::cout << "\n=== Verifying Persisted Data ===" << std::endl;
        
        auto result = executeQuery("SHOW DATABASES");
        if (result && result->buf) {
            std::cout << "Available databases:\n" << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SHOW TABLES FROM mysql_import");
        if (result && result->buf) {
            std::cout << "\nTables in mysql_import:\n" << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.customers");
        if (result && result->buf) {
            std::cout << "\nCustomer count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.orders");
        if (result && result->buf) {
            std::cout << "Order count: " << result->buf << std::endl;
            free_result_v2(result);
        }
    }
    
    void runAnalyticalQueries() {
        std::cout << "\n=== Analytical Queries ===" << std::endl;
        
        // Query 1: Customer count by city
        std::cout << "\n1. Customer count by city:" << std::endl;
        auto result = executeQuery(
            "SELECT city, COUNT(*) as customer_count "
            "FROM mysql_import.customers "
            "GROUP BY city "
            "ORDER BY customer_count DESC");
        if (result && result->buf) {
            std::cout << result->buf << std::endl;
            free_result_v2(result);
        }
        
        // Query 2: Top customers by revenue
        std::cout << "\n2. Top 5 customers by revenue:" << std::endl;
        result = executeQuery(
            "SELECT c.name, SUM(o.price * o.quantity) as total_revenue "
            "FROM mysql_import.customers c "
            "JOIN mysql_import.orders o ON c.id = o.customer_id "
            "GROUP BY c.name "
            "ORDER BY total_revenue DESC "
            "LIMIT 5");
        if (result && result->buf) {
            std::cout << result->buf << std::endl;
            free_result_v2(result);
        }
        
        // Query 3: Pretty format example
        std::cout << "\n3. Customer list (Pretty format):" << std::endl;
        result = executeQuery(
            "SELECT id, name, age, city "
            "FROM mysql_import.customers "
            "ORDER BY id LIMIT 5", "Pretty");
        if (result && result->buf) {
            std::cout << result->buf << std::endl;
            free_result_v2(result);
        }
        
        // Query 4: Monthly statistics
        std::cout << "\n4. Monthly order statistics:" << std::endl;
        result = executeQuery(
            "SELECT toMonth(order_date) as month, "
            "COUNT(*) as orders, "
            "SUM(price * quantity) as revenue "
            "FROM mysql_import.orders "
            "GROUP BY month ORDER BY month");
        if (result && result->buf) {
            std::cout << result->buf << std::endl;
            free_result_v2(result);
        }
    }
};

int main() {
    ClickHouseQuerier querier;
    
    if (!querier.loadChdbLibrary()) {
        std::cerr << "Note: To build libchdb.so, run 'make build' in the chdb directory" << std::endl;
        return 1;
    }
    
    std::cout << "Connecting to persisted data at: " << CHDB_PATH << std::endl;
    
    querier.verifyData();
    querier.runAnalyticalQueries();
    
    std::cout << "\nQuery execution completed!" << std::endl;
    return 0;
}