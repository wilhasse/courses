#include <iostream>
#include <string>
#include <dlfcn.h>
#include <cstring>
#include "common.h"

// Function pointers for chdb
typedef struct chdb_connection_* (*chdb_connect_fn)(int argc, char** argv);
typedef void (*chdb_close_conn_fn)(struct chdb_connection_* conn);
typedef struct chdb_result_* (*chdb_query_fn)(struct chdb_connection_* conn, const char* query, const char* format);
typedef void (*chdb_destroy_query_result_fn)(struct chdb_result_* result);
typedef char* (*chdb_result_buffer_fn)(struct chdb_result_* result);
typedef size_t (*chdb_result_length_fn)(struct chdb_result_* result);
typedef const char* (*chdb_result_error_fn)(struct chdb_result_* result);

// Opaque types
struct chdb_connection_ { void* internal_data; };
struct chdb_result_ { void* internal_data; };

class ClickHouseQuerier {
private:
    void* chdb_handle;
    struct chdb_connection_* conn;
    
    // chdb function pointers
    chdb_connect_fn chdb_connect;
    chdb_close_conn_fn chdb_close_conn;
    chdb_query_fn chdb_query;
    chdb_destroy_query_result_fn chdb_destroy_query_result;
    chdb_result_buffer_fn chdb_result_buffer;
    chdb_result_length_fn chdb_result_length;
    chdb_result_error_fn chdb_result_error;
    
public:
    ClickHouseQuerier() : chdb_handle(nullptr), conn(nullptr) {}
    
    ~ClickHouseQuerier() {
        if (conn && chdb_close_conn) {
            chdb_close_conn(conn);
        }
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool loadChdbLibrary() {
        // Try multiple possible locations for libchdb.so
        const char* lib_paths[] = {
            "/home/cslog/chdb/libchdb.so",
            "../../../../../chdb/libchdb.so",
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
            std::cerr << "Please ensure libchdb.so is built and available" << std::endl;
            return false;
        }
        
        // Load function pointers
        chdb_connect = (chdb_connect_fn)dlsym(chdb_handle, "chdb_connect");
        chdb_close_conn = (chdb_close_conn_fn)dlsym(chdb_handle, "chdb_close_conn");
        chdb_query = (chdb_query_fn)dlsym(chdb_handle, "chdb_query");
        chdb_destroy_query_result = (chdb_destroy_query_result_fn)dlsym(chdb_handle, "chdb_destroy_query_result");
        chdb_result_buffer = (chdb_result_buffer_fn)dlsym(chdb_handle, "chdb_result_buffer");
        chdb_result_length = (chdb_result_length_fn)dlsym(chdb_handle, "chdb_result_length");
        chdb_result_error = (chdb_result_error_fn)dlsym(chdb_handle, "chdb_result_error");
        
        if (!chdb_connect || !chdb_close_conn || !chdb_query || 
            !chdb_destroy_query_result || !chdb_result_buffer || 
            !chdb_result_length || !chdb_result_error) {
            std::cerr << "Failed to load chdb functions: " << dlerror() << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool connect() {
        char path_arg[256];
        snprintf(path_arg, sizeof(path_arg), "--path=%s", CHDB_PATH.c_str());
        
        char* argv[] = {
            (char*)"clickhouse",
            path_arg
        };
        int argc = 2;
        
        conn = chdb_connect(argc, argv);
        if (!conn) {
            std::cerr << "Failed to connect to ClickHouse at: " << CHDB_PATH << std::endl;
            
            // Try with explicit path argument
            char* argv2[] = {
                (char*)"clickhouse",
                (char*)"--path",
                (char*)CHDB_PATH.c_str(),
                nullptr
            };
            conn = chdb_connect(3, argv2);
            
            if (!conn) {
                return false;
            }
        }
        std::cout << "Connected to ClickHouse data at: " << CHDB_PATH << std::endl;
        return true;
    }
    
    void verifyData() {
        std::cout << "\n=== Verifying Persisted Data ===" << std::endl;
        
        // Check if database exists
        struct chdb_result_* result = chdb_query(conn, "SHOW DATABASES", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << "Available databases:\n" << buffer << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Check tables
        result = chdb_query(conn, "SHOW TABLES FROM mysql_import", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << "\nTables in mysql_import database:\n" << buffer << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Count records
        result = chdb_query(conn, "SELECT COUNT(*) as count FROM mysql_import.customers", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << "\nCustomer records: " << buffer << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        result = chdb_query(conn, "SELECT COUNT(*) as count FROM mysql_import.orders", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << "Order records: " << buffer << std::endl;
            }
            chdb_destroy_query_result(result);
        }
    }
    
    void runAnalyticalQueries() {
        std::cout << "\n=== Analytical Queries on Persisted Data ===" << std::endl;
        
        // Query 1: Customer count by city
        std::cout << "\n1. Customer count by city:" << std::endl;
        struct chdb_result_* result = chdb_query(conn, 
            "SELECT city, COUNT(*) as customer_count "
            "FROM mysql_import.customers "
            "GROUP BY city "
            "ORDER BY customer_count DESC", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 2: Total revenue by customer
        std::cout << "\n2. Top 5 customers by revenue:" << std::endl;
        result = chdb_query(conn, 
            "SELECT c.name, SUM(o.price * o.quantity) as total_revenue "
            "FROM mysql_import.customers c "
            "JOIN mysql_import.orders o ON c.id = o.customer_id "
            "GROUP BY c.name "
            "ORDER BY total_revenue DESC "
            "LIMIT 5", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 3: Monthly order statistics
        std::cout << "\n3. Monthly order statistics:" << std::endl;
        result = chdb_query(conn, 
            "SELECT "
            "  toMonth(order_date) as month, "
            "  COUNT(*) as order_count, "
            "  SUM(price * quantity) as total_revenue, "
            "  AVG(price * quantity) as avg_order_value "
            "FROM mysql_import.orders "
            "GROUP BY month "
            "ORDER BY month", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 4: Top selling products
        std::cout << "\n4. Top 5 selling products:" << std::endl;
        result = chdb_query(conn, 
            "SELECT "
            "  product_name, "
            "  SUM(quantity) as total_sold, "
            "  SUM(price * quantity) as total_revenue, "
            "  AVG(price) as avg_price "
            "FROM mysql_import.orders "
            "GROUP BY product_name "
            "ORDER BY total_revenue DESC "
            "LIMIT 5", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 5: Customer age distribution
        std::cout << "\n5. Customer age distribution:" << std::endl;
        result = chdb_query(conn, 
            "SELECT "
            "  CASE "
            "    WHEN age < 30 THEN '< 30' "
            "    WHEN age >= 30 AND age < 40 THEN '30-39' "
            "    WHEN age >= 40 AND age < 50 THEN '40-49' "
            "    ELSE '50+' "
            "  END as age_group, "
            "  COUNT(*) as count, "
            "  AVG(age) as avg_age "
            "FROM mysql_import.customers "
            "GROUP BY age_group "
            "ORDER BY age_group", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 6: Recent orders
        std::cout << "\n6. Recent orders (last 5):" << std::endl;
        result = chdb_query(conn, 
            "SELECT "
            "  o.order_id, "
            "  c.name as customer_name, "
            "  o.product_name, "
            "  o.quantity, "
            "  o.price, "
            "  o.price * o.quantity as total, "
            "  o.order_date "
            "FROM mysql_import.orders o "
            "JOIN mysql_import.customers c ON o.customer_id = c.id "
            "ORDER BY o.order_date DESC, o.order_id DESC "
            "LIMIT 5", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 7: Customer lifetime value
        std::cout << "\n7. Customer lifetime value (customers with 2+ orders):" << std::endl;
        result = chdb_query(conn, 
            "SELECT "
            "  c.name, "
            "  COUNT(DISTINCT o.order_id) as order_count, "
            "  SUM(o.price * o.quantity) as lifetime_value, "
            "  AVG(o.price * o.quantity) as avg_order_value "
            "FROM mysql_import.customers c "
            "JOIN mysql_import.orders o ON c.id = o.customer_id "
            "GROUP BY c.id, c.name "
            "HAVING order_count >= 2 "
            "ORDER BY lifetime_value DESC", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Query 8: Pretty formatted table example
        std::cout << "\n8. Customer summary (Pretty format):" << std::endl;
        result = chdb_query(conn, 
            "SELECT id, name, email, age, city "
            "FROM mysql_import.customers "
            "ORDER BY id "
            "LIMIT 5", "Pretty");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << buffer << std::endl;
            } else {
                std::cerr << "Query error: " << error << std::endl;
            }
            chdb_destroy_query_result(result);
        }
    }
};

int main() {
    ClickHouseQuerier querier;
    
    // Load chdb library
    if (!querier.loadChdbLibrary()) {
        std::cerr << "Note: To build libchdb.so, run 'make build' in the chdb directory" << std::endl;
        return 1;
    }
    
    // Connect to persisted ClickHouse data
    if (!querier.connect()) {
        std::cerr << "Failed to connect to ClickHouse data. Make sure feed_data was run first." << std::endl;
        return 1;
    }
    
    // Verify data exists
    querier.verifyData();
    
    // Run analytical queries
    querier.runAnalyticalQueries();
    
    std::cout << "\nQuery execution completed successfully!" << std::endl;
    return 0;
}