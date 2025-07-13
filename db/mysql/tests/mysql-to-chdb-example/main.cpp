#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
#include "chdb.h"

struct Customer {
    int id;
    std::string name;
    std::string email;
    int age;
    std::string city;
    std::string created_at;
};

struct Order {
    int order_id;
    int customer_id;
    std::string product_name;
    int quantity;
    double price;
    std::string order_date;
};

class MySQLToClickHouse {
private:
    MYSQL* mysql_conn;
    
public:
    MySQLToClickHouse() : mysql_conn(nullptr) {}
    
    ~MySQLToClickHouse() {
        if (mysql_conn) {
            mysql_close(mysql_conn);
        }
    }
    
    bool connectToMySQL(const std::string& host, const std::string& user, 
                       const std::string& password, const std::string& database) {
        mysql_conn = mysql_init(nullptr);
        if (!mysql_conn) {
            std::cerr << "MySQL init failed" << std::endl;
            return false;
        }
        
        if (!mysql_real_connect(mysql_conn, host.c_str(), user.c_str(), 
                               password.c_str(), database.c_str(), 0, nullptr, 0)) {
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
        // Create tables in ClickHouse
        std::string create_customers_table = R"(
            CREATE TABLE IF NOT EXISTS customers (
                id Int32,
                name String,
                email String,
                age Int32,
                city String,
                created_at DateTime
            ) ENGINE = Memory
        )";
        
        std::string create_orders_table = R"(
            CREATE TABLE IF NOT EXISTS orders (
                order_id Int32,
                customer_id Int32,
                product_name String,
                quantity Int32,
                price Float64,
                order_date Date
            ) ENGINE = Memory
        )";
        
        // Execute create table statements
        chdb_conn* conn = chdb_connect();
        
        chdb_result* result = chdb_query(conn, create_customers_table.c_str());
        if (result) {
            std::cout << "Customers table created in ClickHouse" << std::endl;
            chdb_free_result(result);
        }
        
        result = chdb_query(conn, create_orders_table.c_str());
        if (result) {
            std::cout << "Orders table created in ClickHouse" << std::endl;
            chdb_free_result(result);
        }
        
        // Insert customers data
        for (const auto& customer : customers) {
            std::stringstream ss;
            ss << "INSERT INTO customers VALUES ("
               << customer.id << ", "
               << "'" << customer.name << "', "
               << "'" << customer.email << "', "
               << customer.age << ", "
               << "'" << customer.city << "', "
               << "'" << customer.created_at << "')";
            
            result = chdb_query(conn, ss.str().c_str());
            if (result) {
                chdb_free_result(result);
            }
        }
        std::cout << "Loaded " << customers.size() << " customers to ClickHouse" << std::endl;
        
        // Insert orders data
        for (const auto& order : orders) {
            std::stringstream ss;
            ss << "INSERT INTO orders VALUES ("
               << order.order_id << ", "
               << order.customer_id << ", "
               << "'" << order.product_name << "', "
               << order.quantity << ", "
               << order.price << ", "
               << "'" << order.order_date << "')";
            
            result = chdb_query(conn, ss.str().c_str());
            if (result) {
                chdb_free_result(result);
            }
        }
        std::cout << "Loaded " << orders.size() << " orders to ClickHouse" << std::endl;
        
        chdb_disconnect(conn);
    }
    
    void runSampleQueries() {
        chdb_conn* conn = chdb_connect();
        
        std::cout << "\n=== Sample Queries ===" << std::endl;
        
        // Query 1: Count customers by city
        std::cout << "\n1. Customer count by city:" << std::endl;
        chdb_result* result = chdb_query(conn, 
            "SELECT city, COUNT(*) as customer_count FROM customers GROUP BY city ORDER BY customer_count DESC");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Query 2: Total revenue by customer
        std::cout << "\n2. Total revenue by customer:" << std::endl;
        result = chdb_query(conn, 
            "SELECT c.name, SUM(o.price * o.quantity) as total_revenue "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.name ORDER BY total_revenue DESC LIMIT 5");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Query 3: Average order value by month
        std::cout << "\n3. Average order value by month:" << std::endl;
        result = chdb_query(conn, 
            "SELECT toMonth(order_date) as month, AVG(price * quantity) as avg_order_value "
            "FROM orders GROUP BY month ORDER BY month");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Query 4: Top selling products
        std::cout << "\n4. Top selling products:" << std::endl;
        result = chdb_query(conn, 
            "SELECT product_name, SUM(quantity) as total_sold, SUM(price * quantity) as total_revenue "
            "FROM orders GROUP BY product_name ORDER BY total_revenue DESC LIMIT 5");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Query 5: Customer age distribution
        std::cout << "\n5. Customer age distribution:" << std::endl;
        result = chdb_query(conn, 
            "SELECT CASE "
            "WHEN age < 30 THEN '< 30' "
            "WHEN age >= 30 AND age < 40 THEN '30-39' "
            "WHEN age >= 40 AND age < 50 THEN '40-49' "
            "ELSE '50+' END as age_group, "
            "COUNT(*) as count FROM customers GROUP BY age_group ORDER BY age_group");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        chdb_disconnect(conn);
    }
};

int main() {
    MySQLToClickHouse converter;
    
    // Connect to MySQL
    if (!converter.connectToMySQL("localhost", "root", "teste", "sample_db")) {
        return 1;
    }
    
    // Fetch data from MySQL
    std::cout << "\nFetching data from MySQL..." << std::endl;
    std::vector<Customer> customers = converter.fetchCustomers();
    std::vector<Order> orders = converter.fetchOrders();
    
    std::cout << "Fetched " << customers.size() << " customers" << std::endl;
    std::cout << "Fetched " << orders.size() << " orders" << std::endl;
    
    // Load data into ClickHouse using chdb
    std::cout << "\nLoading data into ClickHouse..." << std::endl;
    converter.loadToClickHouse(customers, orders);
    
    // Run sample queries
    converter.runSampleQueries();
    
    return 0;
}