#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
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

class MySQLToClickHouseFeeder {
private:
    MYSQL* mysql_conn;
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
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
        const char* lib_paths[] = {
            "/usr/local/lib/libchdb.so",
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
        std::cout << "Loading data into ClickHouse at path: " << CHDB_PATH << std::endl;
        
        // Create database
        auto result = executeQuery("CREATE DATABASE IF NOT EXISTS mysql_import");
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating database: " << result->error_message << std::endl;
            } else {
                std::cout << "Database created/verified" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Create customers table
        std::string create_customers = R"(
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
        
        result = executeQuery(create_customers);
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating customers table: " << result->error_message << std::endl;
            } else {
                std::cout << "Customers table created" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Create orders table
        std::string create_orders = R"(
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
        
        result = executeQuery(create_orders);
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating orders table: " << result->error_message << std::endl;
            } else {
                std::cout << "Orders table created" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Clear existing data
        executeQuery("TRUNCATE TABLE IF EXISTS mysql_import.customers");
        executeQuery("TRUNCATE TABLE IF EXISTS mysql_import.orders");
        
        // Insert customers
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
            
            result = executeQuery(ss.str());
            if (result) {
                if (!result->error_message) {
                    inserted_customers++;
                }
                free_result_v2(result);
            }
        }
        std::cout << "Inserted " << inserted_customers << " customers" << std::endl;
        
        // Insert orders
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
            
            result = executeQuery(ss.str());
            if (result) {
                if (!result->error_message) {
                    inserted_orders++;
                }
                free_result_v2(result);
            }
        }
        std::cout << "Inserted " << inserted_orders << " orders" << std::endl;
        
        // Verify counts
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.customers");
        if (result && result->buf) {
            std::cout << "Final customer count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.orders");
        if (result && result->buf) {
            std::cout << "Final order count: " << result->buf << std::endl;
            free_result_v2(result);
        }
    }
};

int main() {
    MySQLToClickHouseFeeder feeder;
    
    if (!feeder.loadChdbLibrary()) {
        std::cerr << "Note: To build libchdb.so, run 'make build' in the chdb directory" << std::endl;
        return 1;
    }
    
    if (!feeder.connectToMySQL()) {
        return 1;
    }
    
    std::cout << "\nFetching data from MySQL..." << std::endl;
    std::vector<Customer> customers = feeder.fetchCustomers();
    std::vector<Order> orders = feeder.fetchOrders();
    
    std::cout << "Fetched " << customers.size() << " customers" << std::endl;
    std::cout << "Fetched " << orders.size() << " orders" << std::endl;
    
    std::cout << "\nLoading data into ClickHouse..." << std::endl;
    feeder.loadToClickHouse(customers, orders);
    
    std::cout << "\nData feeding completed!" << std::endl;
    return 0;
}