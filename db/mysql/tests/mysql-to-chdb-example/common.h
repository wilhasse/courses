#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>

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

// Database path for persistence
const std::string CHDB_PATH = "./clickhouse_data";

// MySQL connection parameters
const std::string MYSQL_HOST = "localhost";
const std::string MYSQL_USER = "root";
const std::string MYSQL_PASSWORD = "teste";
const std::string MYSQL_DATABASE = "sample_db";

#endif // COMMON_H