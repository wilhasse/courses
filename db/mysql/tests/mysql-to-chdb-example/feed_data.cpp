#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
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

class MySQLToClickHouseFeeder {
private:
    MYSQL* mysql_conn;
    void* chdb_handle;
    
    // chdb function pointers
    chdb_connect_fn chdb_connect;
    chdb_close_conn_fn chdb_close_conn;
    chdb_query_fn chdb_query;
    chdb_destroy_query_result_fn chdb_destroy_query_result;
    chdb_result_buffer_fn chdb_result_buffer;
    chdb_result_length_fn chdb_result_length;
    chdb_result_error_fn chdb_result_error;
    
public:
    MySQLToClickHouseFeeder() : mysql_conn(nullptr), chdb_handle(nullptr) {}
    
    ~MySQLToClickHouseFeeder() {
        if (mysql_conn) {
            mysql_close(mysql_conn);
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
    
    bool connectToMySQL() {
        mysql_conn = mysql_init(nullptr);
        if (!mysql_conn) {
            std::cerr << "MySQL init failed" << std::endl;
            return false;
        }
        
        if (!mysql_real_connect(mysql_conn, MYSQL_HOST.c_str(), MYSQL_USER.c_str(), 
                               MYSQL_PASSWORD.c_str(), MYSQL_DATABASE.c_str(), 0, nullptr, 0)) {
            std::cerr << "MySQL connection failed: " << mysql_error(mysql_conn) << std::endl;
            return false;
        }
        
        std::cout << "Connected to MySQL successfully!" << std::endl;
        return true;
    }
    
    std::vector<Customer> fetchCustomers() {
        std::vector<Customer> customers;
        
        if (mysql_query(mysql_conn, "SELECT id, name, email, age, city, created_at FROM customers")) {
            std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
            return customers;
        }
        
        MYSQL_RES* result = mysql_store_result(mysql_conn);
        if (!result) {
            std::cerr << "MySQL store result failed: " << mysql_error(mysql_conn) << std::endl;
            return customers;
        }
        
        MYSQL_ROW row;
        while ((row = mysql_fetch_row(result))) {
            Customer customer;
            customer.id = std::stoi(row[0] ? row[0] : "0");
            customer.name = row[1] ? row[1] : "";
            customer.email = row[2] ? row[2] : "";
            customer.age = std::stoi(row[3] ? row[3] : "0");
            customer.city = row[4] ? row[4] : "";
            customer.created_at = row[5] ? row[5] : "";
            customers.push_back(customer);
        }
        
        mysql_free_result(result);
        return customers;
    }
    
    std::vector<Order> fetchOrders() {
        std::vector<Order> orders;
        
        if (mysql_query(mysql_conn, "SELECT order_id, customer_id, product_name, quantity, price, order_date FROM orders")) {
            std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
            return orders;
        }
        
        MYSQL_RES* result = mysql_store_result(mysql_conn);
        if (!result) {
            std::cerr << "MySQL store result failed: " << mysql_error(mysql_conn) << std::endl;
            return orders;
        }
        
        MYSQL_ROW row;
        while ((row = mysql_fetch_row(result))) {
            Order order;
            order.order_id = std::stoi(row[0] ? row[0] : "0");
            order.customer_id = std::stoi(row[1] ? row[1] : "0");
            order.product_name = row[2] ? row[2] : "";
            order.quantity = std::stoi(row[3] ? row[3] : "0");
            order.price = std::stod(row[4] ? row[4] : "0.0");
            order.order_date = row[5] ? row[5] : "";
            orders.push_back(order);
        }
        
        mysql_free_result(result);
        return orders;
    }
    
    void loadToClickHouse(const std::vector<Customer>& customers, const std::vector<Order>& orders) {
        // Connect to chdb with persistence
        char path_arg[256];
        snprintf(path_arg, sizeof(path_arg), "--path=%s", CHDB_PATH.c_str());
        
        char* argv[] = {
            (char*)"clickhouse",
            path_arg
        };
        int argc = 2;
        
        struct chdb_connection_* conn = chdb_connect(argc, argv);
        if (!conn) {
            std::cerr << "Failed to connect to chdb" << std::endl;
            std::cerr << "Trying with different arguments..." << std::endl;
            
            // Try with explicit path argument
            char* argv2[] = {
                (char*)"clickhouse",
                (char*)"--path",
                (char*)CHDB_PATH.c_str(),
                nullptr
            };
            conn = chdb_connect(3, argv2);
            
            if (!conn) {
                std::cerr << "Still failed. Trying with just path..." << std::endl;
                // Try with just the path
                char* argv3[] = {
                    (char*)CHDB_PATH.c_str(),
                    nullptr
                };
                conn = chdb_connect(1, argv3);
                
                if (!conn) {
                    std::cerr << "Connection failed with all attempts" << std::endl;
                    return;
                }
            }
        }
        
        std::cout << "Connected to chdb with path: " << CHDB_PATH << std::endl;
        
        // Create database
        std::string create_db = "CREATE DATABASE IF NOT EXISTS mysql_import";
        struct chdb_result_* result = chdb_query(conn, create_db.c_str(), "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (error && strlen(error) > 0) {
                std::cerr << "Error creating database: " << error << std::endl;
            } else {
                std::cout << "Database created/verified in ClickHouse" << std::endl;
                // Also print the result to see what we get
                char* buffer = chdb_result_buffer(result);
                if (buffer && strlen(buffer) > 0) {
                    std::cout << "Result: " << buffer << std::endl;
                }
            }
            chdb_destroy_query_result(result);
        } else {
            std::cerr << "Query returned NULL result" << std::endl;
        }
        
        // Create tables
        std::string create_customers_table = R"(
            CREATE TABLE IF NOT EXISTS mysql_import.customers (
                id Int32,
                name String,
                email String,
                age Int32,
                city String,
                created_at DateTime
            ) ENGINE = MergeTree()
            ORDER BY id
        )";
        
        result = chdb_query(conn, create_customers_table.c_str(), "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (error) {
                std::cerr << "Error creating customers table: " << error << std::endl;
            } else {
                std::cout << "Customers table created in ClickHouse" << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        std::string create_orders_table = R"(
            CREATE TABLE IF NOT EXISTS mysql_import.orders (
                order_id Int32,
                customer_id Int32,
                product_name String,
                quantity Int32,
                price Float64,
                order_date Date
            ) ENGINE = MergeTree()
            ORDER BY order_id
        )";
        
        result = chdb_query(conn, create_orders_table.c_str(), "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (error) {
                std::cerr << "Error creating orders table: " << error << std::endl;
            } else {
                std::cout << "Orders table created in ClickHouse" << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        // Clear existing data
        result = chdb_query(conn, "TRUNCATE TABLE mysql_import.customers", "CSV");
        if (result) chdb_destroy_query_result(result);
        
        result = chdb_query(conn, "TRUNCATE TABLE mysql_import.orders", "CSV");
        if (result) chdb_destroy_query_result(result);
        
        // Insert customers data
        int inserted_customers = 0;
        for (const auto& customer : customers) {
            std::stringstream ss;
            ss << "INSERT INTO mysql_import.customers VALUES ("
               << customer.id << ", "
               << "'" << customer.name << "', "
               << "'" << customer.email << "', "
               << customer.age << ", "
               << "'" << customer.city << "', "
               << "'" << customer.created_at << "')";
            
            result = chdb_query(conn, ss.str().c_str(), "CSV");
            if (result) {
                const char* error = chdb_result_error(result);
                if (!error) {
                    inserted_customers++;
                }
                chdb_destroy_query_result(result);
            }
        }
        std::cout << "Loaded " << inserted_customers << " customers to ClickHouse" << std::endl;
        
        // Insert orders data
        int inserted_orders = 0;
        for (const auto& order : orders) {
            std::stringstream ss;
            ss << "INSERT INTO mysql_import.orders VALUES ("
               << order.order_id << ", "
               << order.customer_id << ", "
               << "'" << order.product_name << "', "
               << order.quantity << ", "
               << order.price << ", "
               << "'" << order.order_date << "')";
            
            result = chdb_query(conn, ss.str().c_str(), "CSV");
            if (result) {
                const char* error = chdb_result_error(result);
                if (!error) {
                    inserted_orders++;
                }
                chdb_destroy_query_result(result);
            }
        }
        std::cout << "Loaded " << inserted_orders << " orders to ClickHouse" << std::endl;
        
        // Verify data was inserted
        result = chdb_query(conn, "SELECT COUNT(*) FROM mysql_import.customers", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << "Customer count in ClickHouse: " << buffer << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        result = chdb_query(conn, "SELECT COUNT(*) FROM mysql_import.orders", "CSV");
        if (result) {
            const char* error = chdb_result_error(result);
            if (!error) {
                char* buffer = chdb_result_buffer(result);
                std::cout << "Order count in ClickHouse: " << buffer << std::endl;
            }
            chdb_destroy_query_result(result);
        }
        
        chdb_close_conn(conn);
        std::cout << "\nData persisted to: " << CHDB_PATH << std::endl;
    }
};

int main() {
    MySQLToClickHouseFeeder feeder;
    
    // Load chdb library
    if (!feeder.loadChdbLibrary()) {
        std::cerr << "Note: To build libchdb.so, run 'make build' in the chdb directory" << std::endl;
        return 1;
    }
    
    // Connect to MySQL
    if (!feeder.connectToMySQL()) {
        return 1;
    }
    
    // Fetch data from MySQL
    std::cout << "\nFetching data from MySQL..." << std::endl;
    std::vector<Customer> customers = feeder.fetchCustomers();
    std::vector<Order> orders = feeder.fetchOrders();
    
    std::cout << "Fetched " << customers.size() << " customers from MySQL" << std::endl;
    std::cout << "Fetched " << orders.size() << " orders from MySQL" << std::endl;
    
    // Load data into ClickHouse using chdb
    std::cout << "\nLoading data into ClickHouse..." << std::endl;
    feeder.loadToClickHouse(customers, orders);
    
    std::cout << "\nData feeding completed successfully!" << std::endl;
    return 0;
}