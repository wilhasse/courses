#include <iostream>
#include <string>
#include "chdb_persist.h"
#include "common.h"

class ClickHouseQuerier {
private:
    chdb_conn* conn;
    
public:
    ClickHouseQuerier() : conn(nullptr) {}
    
    ~ClickHouseQuerier() {
        if (conn) {
            chdb_disconnect(conn);
        }
    }
    
    bool connect() {
        conn = chdb_connect(CHDB_PATH.c_str());
        if (!conn) {
            std::cerr << "Failed to connect to ClickHouse at: " << CHDB_PATH << std::endl;
            return false;
        }
        std::cout << "Connected to ClickHouse data at: " << CHDB_PATH << std::endl;
        return true;
    }
    
    void verifyData() {
        std::cout << "\n=== Verifying Persisted Data ===" << std::endl;
        
        // Check if database exists
        chdb_result* result = chdb_query(conn, "SHOW DATABASES");
        if (result) {
            std::cout << "Available databases:\n" << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Check tables
        result = chdb_query(conn, "SHOW TABLES FROM mysql_import");
        if (result) {
            std::cout << "\nTables in mysql_import database:\n" << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Count records
        result = chdb_query(conn, "SELECT COUNT(*) as count FROM mysql_import.customers");
        if (result) {
            std::cout << "\nCustomer records: " << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        result = chdb_query(conn, "SELECT COUNT(*) as count FROM mysql_import.orders");
        if (result) {
            std::cout << "Order records: " << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
    }
    
    void runAnalyticalQueries() {
        std::cout << "\n=== Analytical Queries on Persisted Data ===" << std::endl;
        
        // Query 1: Customer count by city
        std::cout << "\n1. Customer count by city:" << std::endl;
        chdb_result* result = chdb_query(conn, 
            "SELECT city, COUNT(*) as customer_count "
            "FROM mysql_import.customers "
            "GROUP BY city "
            "ORDER BY customer_count DESC");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
        
        // Query 2: Total revenue by customer
        std::cout << "\n2. Top 5 customers by revenue:" << std::endl;
        result = chdb_query(conn, 
            "SELECT c.name, SUM(o.price * o.quantity) as total_revenue "
            "FROM mysql_import.customers c "
            "JOIN mysql_import.orders o ON c.id = o.customer_id "
            "GROUP BY c.name "
            "ORDER BY total_revenue DESC "
            "LIMIT 5");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
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
            "ORDER BY month");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
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
            "LIMIT 5");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
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
            "ORDER BY age_group");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
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
            "LIMIT 5");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
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
            "ORDER BY lifetime_value DESC");
        if (result) {
            std::cout << chdb_result_to_string(result) << std::endl;
            chdb_free_result(result);
        }
    }
};

int main() {
    ClickHouseQuerier querier;
    
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