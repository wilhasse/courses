/**
 * MySQL UDF functions to simulate table-valued functions for ClickHouse data
 * Allows joining ClickHouse tables with MySQL tables
 */

#include <mysql.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>

// Configuration
const char* CHDB_API_HOST = "127.0.0.1";
const int CHDB_API_PORT = 8125;

// Cache for table metadata
static std::map<std::string, long long> table_row_counts;

extern "C" {

// Helper function to query API server
std::string query_api_server(const std::string& query) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return "";
    
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(CHDB_API_PORT);
    inet_pton(AF_INET, CHDB_API_HOST, &server_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        return "";
    }
    
    // Send query
    uint32_t size = htonl(query.size());
    write(sock, &size, 4);
    write(sock, query.c_str(), query.size());
    
    // Read response
    uint32_t response_size;
    if (read(sock, &response_size, 4) != 4) {
        close(sock);
        return "";
    }
    response_size = ntohl(response_size);
    
    if (response_size == 0 || response_size > 1048576) {
        close(sock);
        return "";
    }
    
    std::vector<char> buffer(response_size);
    size_t total_read = 0;
    while (total_read < response_size) {
        ssize_t n = read(sock, buffer.data() + total_read, response_size - total_read);
        if (n <= 0) {
            close(sock);
            return "";
        }
        total_read += n;
    }
    
    close(sock);
    
    std::string result(buffer.data(), response_size);
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }
    
    return result;
}

// Parse TSV row into fields
std::vector<std::string> parse_tsv_row(const std::string& row) {
    std::vector<std::string> fields;
    std::stringstream ss(row);
    std::string field;
    
    while (std::getline(ss, field, '\t')) {
        fields.push_back(field);
    }
    
    return fields;
}

// ========== Row Count Functions ==========

// chdb_table_row_count(table_name) - returns number of rows in ClickHouse table
bool chdb_table_row_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_table_row_count() requires table name");
        return 1;
    }
    args->arg_type[0] = STRING_RESULT;
    initid->maybe_null = 1;
    return 0;
}

void chdb_table_row_count_deinit(UDF_INIT *initid) {}

long long chdb_table_row_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return 0;
    }
    
    std::string table(args->args[0], args->lengths[0]);
    
    // Check cache
    auto it = table_row_counts.find(table);
    if (it != table_row_counts.end()) {
        return it->second;
    }
    
    // Query row count
    std::string query = "SELECT COUNT(*) FROM " + table;
    std::string result = query_api_server(query);
    
    if (result.empty()) {
        *is_null = 1;
        return 0;
    }
    
    try {
        long long count = std::stoll(result);
        table_row_counts[table] = count;  // Cache the result
        return count;
    } catch (...) {
        *is_null = 1;
        return 0;
    }
}

// ========== Field Access Functions ==========

// Buffer for string results
static thread_local char result_buffer[65536];

// chdb_table_get_field(table, field, row_num) - get field value for specific row
bool chdb_table_get_field_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 3) {
        strcpy(message, "chdb_table_get_field(table, field, row_num) requires 3 arguments");
        return 1;
    }
    args->arg_type[0] = STRING_RESULT;  // table
    args->arg_type[1] = STRING_RESULT;  // field
    args->arg_type[2] = INT_RESULT;     // row_num
    initid->maybe_null = 1;
    initid->max_length = sizeof(result_buffer) - 1;
    return 0;
}

void chdb_table_get_field_deinit(UDF_INIT *initid) {}

char* chdb_table_get_field(UDF_INIT *initid, UDF_ARGS *args, char *result,
                          unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1] || !args->args[2]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string table(args->args[0], args->lengths[0]);
    std::string field(args->args[1], args->lengths[1]);
    long long row_num = *((long long*)args->args[2]);
    
    if (row_num < 1) {
        *is_null = 1;
        return NULL;
    }
    
    // Query specific row
    std::stringstream query;
    query << "SELECT " << field << " FROM " << table 
          << " ORDER BY 1 LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result_str = query_api_server(query.str());
    
    if (result_str.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(result_buffer, result_str.c_str(), sizeof(result_buffer) - 1);
    result_buffer[sizeof(result_buffer) - 1] = '\0';
    *length = strlen(result_buffer);
    return result_buffer;
}

// ========== Row Retrieval Functions ==========

// chdb_table_get_row(table, row_num, format) - get entire row as formatted string
bool chdb_table_get_row_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count < 2 || args->arg_count > 3) {
        strcpy(message, "chdb_table_get_row(table, row_num, [format]) requires 2-3 arguments");
        return 1;
    }
    args->arg_type[0] = STRING_RESULT;  // table
    args->arg_type[1] = INT_RESULT;     // row_num
    if (args->arg_count == 3) {
        args->arg_type[2] = STRING_RESULT;  // format (optional)
    }
    initid->maybe_null = 1;
    initid->max_length = sizeof(result_buffer) - 1;
    return 0;
}

void chdb_table_get_row_deinit(UDF_INIT *initid) {}

char* chdb_table_get_row(UDF_INIT *initid, UDF_ARGS *args, char *result,
                        unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string table(args->args[0], args->lengths[0]);
    long long row_num = *((long long*)args->args[1]);
    std::string format = "TabSeparated";
    
    if (args->arg_count == 3 && args->args[2]) {
        format = std::string(args->args[2], args->lengths[2]);
    }
    
    if (row_num < 1) {
        *is_null = 1;
        return NULL;
    }
    
    // Query specific row with all fields
    std::stringstream query;
    query << "SELECT * FROM " << table 
          << " ORDER BY 1 LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result_str = query_api_server(query.str());
    
    if (result_str.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(result_buffer, result_str.c_str(), sizeof(result_buffer) - 1);
    result_buffer[sizeof(result_buffer) - 1] = '\0';
    *length = strlen(result_buffer);
    return result_buffer;
}

// ========== Specialized Customer Table Functions ==========

// chdb_customers_get_id(row_num) - get customer ID for given row
bool chdb_customers_get_id_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_customers_get_id() requires row number");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    return 0;
}

void chdb_customers_get_id_deinit(UDF_INIT *initid) {}

long long chdb_customers_get_id(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return 0;
    }
    
    std::stringstream query;
    query << "SELECT id FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result = query_api_server(query.str());
    if (result.empty()) {
        *is_null = 1;
        return 0;
    }
    
    try {
        return std::stoll(result);
    } catch (...) {
        *is_null = 1;
        return 0;
    }
}

// chdb_customers_get_name(row_num) - get customer name for given row
bool chdb_customers_get_name_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_customers_get_name() requires row number");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 255;
    return 0;
}

void chdb_customers_get_name_deinit(UDF_INIT *initid) {}

char* chdb_customers_get_name(UDF_INIT *initid, UDF_ARGS *args, char *result,
                             unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return NULL;
    }
    
    std::stringstream query;
    query << "SELECT name FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result_str = query_api_server(query.str());
    if (result_str.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(result_buffer, result_str.c_str(), sizeof(result_buffer) - 1);
    result_buffer[sizeof(result_buffer) - 1] = '\0';
    *length = strlen(result_buffer);
    return result_buffer;
}

// chdb_customers_get_city(row_num) - get customer city for given row
bool chdb_customers_get_city_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_customers_get_city() requires row number");
        return 1;
    }
    args->arg_type[0] = INT_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 255;
    return 0;
}

void chdb_customers_get_city_deinit(UDF_INIT *initid) {}

char* chdb_customers_get_city(UDF_INIT *initid, UDF_ARGS *args, char *result,
                             unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    long long row_num = *((long long*)args->args[0]);
    if (row_num < 1) {
        *is_null = 1;
        return NULL;
    }
    
    std::stringstream query;
    query << "SELECT city FROM mysql_import.customers ORDER BY id LIMIT 1 OFFSET " << (row_num - 1);
    
    std::string result_str = query_api_server(query.str());
    if (result_str.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(result_buffer, result_str.c_str(), sizeof(result_buffer) - 1);
    result_buffer[sizeof(result_buffer) - 1] = '\0';
    *length = strlen(result_buffer);
    return result_buffer;
}

} // extern "C"