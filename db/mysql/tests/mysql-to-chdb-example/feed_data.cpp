#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
#include "chdb_persist.h"
#include "common.h"

class MySQLToClickHouseFeeder {
private:
    MYSQL* mysql_conn;
    
public:
    MySQLToClickHouseFeeder() : mysql_conn(nullptr) {}
    
    ~MySQLToClickHouseFeeder() {
        if (mysql_conn) {
            mysql_close(mysql_conn);
        }
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
        chdb_conn* conn = chdb_connect(CHDB_PATH.c_str());
        
        // Create database
        std::string create_db = "CREATE DATABASE IF NOT EXISTS mysql_import";
        chdb_result* result = chdb_query(conn, create_db.c_str());
        if (result) {
            std::cout << "Database created/verified in ClickHouse" << std::endl;
            chdb_free_result(result);
        }
        
        // Create tables in the database
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
        
        // Execute create table statements
        result = chdb_query(conn, create_customers_table.c_str());
        if (result) {
            std::cout << "Customers table created in ClickHouse" << std::endl;
            chdb_free_result(result);
        }
        
        result = chdb_query(conn, create_orders_table.c_str());
        if (result) {
            std::cout << "Orders table created in ClickHouse" << std::endl;
            chdb_free_result(result);
        }
        
        // Clear existing data
        result = chdb_query(conn, "TRUNCATE TABLE mysql_import.customers");
        if (result) chdb_free_result(result);
        
        result = chdb_query(conn, "TRUNCATE TABLE mysql_import.orders");
        if (result) chdb_free_result(result);
        
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
            
            result = chdb_query(conn, ss.str().c_str());
            if (result) {
                chdb_free_result(result);
                inserted_customers++;
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
            
            result = chdb_query(conn, ss.str().c_str());
            if (result) {
                chdb_free_result(result);
                inserted_orders++;
            }
        }
        std::cout << "Loaded " << inserted_orders << " orders to ClickHouse" << std::endl;
        
        // Verify data was inserted
        result = chdb_query(conn, "SELECT COUNT(*) FROM mysql_import.customers");
        if (result) {
            std::cout << "Customer count in ClickHouse: " << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        result = chdb_query(conn, "SELECT COUNT(*) FROM mysql_import.orders");
        if (result) {
            std::cout << "Order count in ClickHouse: " << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        chdb_disconnect(conn);
        std::cout << "\nData persisted to: " << CHDB_PATH << std::endl;
    }
};

int main() {
    MySQLToClickHouseFeeder feeder;
    
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