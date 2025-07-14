/**
 * MySQL UDF functions that return JSON for use with JSON_TABLE
 * This enables true table-valued function behavior in MySQL 8.0.19+
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

// Configuration
const char* CHDB_API_HOST = "127.0.0.1";
const int CHDB_API_PORT = 8125;

// Large buffer for JSON results
static thread_local char json_buffer[1048576];  // 1MB buffer

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
    
    if (response_size == 0 || response_size > sizeof(json_buffer) - 1) {
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

// Parse TSV to JSON array
std::string tsv_to_json(const std::string& tsv_data, const std::vector<std::string>& column_names) {
    std::stringstream json;
    json << "[";
    
    std::stringstream ss(tsv_data);
    std::string line;
    bool first_row = true;
    
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        
        if (!first_row) json << ",";
        first_row = false;
        
        json << "{";
        
        std::stringstream line_ss(line);
        std::string field;
        size_t col_idx = 0;
        bool first_field = true;
        
        while (std::getline(line_ss, field, '\t') && col_idx < column_names.size()) {
            if (!first_field) json << ",";
            first_field = false;
            
            json << "\"" << column_names[col_idx] << "\":";
            
            // Try to parse as number
            bool is_number = true;
            bool has_decimal = false;
            for (char c : field) {
                if (c == '.') {
                    if (has_decimal) {
                        is_number = false;
                        break;
                    }
                    has_decimal = true;
                } else if (!isdigit(c) && c != '-' && c != '+') {
                    is_number = false;
                    break;
                }
            }
            
            if (is_number && !field.empty()) {
                json << field;
            } else {
                // Escape quotes in string
                json << "\"";
                for (char c : field) {
                    if (c == '"') json << "\\\"";
                    else if (c == '\\') json << "\\\\";
                    else json << c;
                }
                json << "\"";
            }
            
            col_idx++;
        }
        
        json << "}";
    }
    
    json << "]";
    return json.str();
}

// ========== Main JSON Table Function ==========

// chdb_table_json(query, columns) - Execute query and return JSON for JSON_TABLE
bool chdb_table_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 2) {
        strcpy(message, "chdb_table_json(query, columns) requires 2 arguments");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT || args->arg_type[1] != STRING_RESULT) {
        strcpy(message, "Both arguments must be strings");
        return 1;
    }
    
    initid->maybe_null = 1;
    initid->max_length = sizeof(json_buffer) - 1;
    initid->const_item = 0;
    // Force string result to be treated as UTF-8
    initid->ptr = (char*)1; // Non-null to indicate string result
    return 0;
}

void chdb_table_json_deinit(UDF_INIT *initid) {}

char* chdb_table_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                     unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    std::string columns_str(args->args[1], args->lengths[1]);
    
    // Parse column names
    std::vector<std::string> columns;
    std::stringstream ss(columns_str);
    std::string col;
    while (std::getline(ss, col, ',')) {
        // Trim whitespace
        col.erase(0, col.find_first_not_of(" \t"));
        col.erase(col.find_last_not_of(" \t") + 1);
        columns.push_back(col);
    }
    
    // Execute query
    std::string tsv_result = query_api_server(query);
    if (tsv_result.empty()) {
        strcpy(json_buffer, "[]");
        *length = 2;
        return json_buffer;
    }
    
    // Convert to JSON
    std::string json_result = tsv_to_json(tsv_result, columns);
    
    // Copy to buffer
    strncpy(json_buffer, json_result.c_str(), sizeof(json_buffer) - 1);
    json_buffer[sizeof(json_buffer) - 1] = '\0';
    *length = strlen(json_buffer);
    
    return json_buffer;
}

// ========== Specialized Functions ==========

// chdb_customers_json() - Get all customers as JSON
bool chdb_customers_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 0) {
        strcpy(message, "chdb_customers_json() takes no arguments");
        return 1;
    }
    
    initid->maybe_null = 1;
    initid->max_length = sizeof(json_buffer) - 1;
    initid->const_item = 0;
    // Force string result to be treated as UTF-8
    initid->ptr = (char*)1; // Non-null to indicate string result
    return 0;
}

void chdb_customers_json_deinit(UDF_INIT *initid) {}

char* chdb_customers_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                         unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    // Query all customers
    std::string query = "SELECT id, name, email, age, city FROM mysql_import.customers ORDER BY id";
    std::string tsv_result = query_api_server(query);
    
    if (tsv_result.empty()) {
        strcpy(json_buffer, "[]");
        *length = 2;
        return json_buffer;
    }
    
    // Define columns
    std::vector<std::string> columns = {"id", "name", "email", "age", "city"};
    
    // Convert to JSON
    std::string json_result = tsv_to_json(tsv_result, columns);
    
    // Copy to buffer
    strncpy(json_buffer, json_result.c_str(), sizeof(json_buffer) - 1);
    json_buffer[sizeof(json_buffer) - 1] = '\0';
    *length = strlen(json_buffer);
    
    return json_buffer;
}

// chdb_query_json(query) - Execute query and return JSON with auto-detected columns
bool chdb_query_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_query_json(query) requires 1 argument");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT) {
        strcpy(message, "Argument must be a string");
        return 1;
    }
    
    initid->maybe_null = 1;
    initid->max_length = sizeof(json_buffer) - 1;
    initid->const_item = 0;
    // Force string result to be treated as UTF-8
    initid->ptr = (char*)1; // Non-null to indicate string result
    return 0;
}

void chdb_query_json_deinit(UDF_INIT *initid) {}

char* chdb_query_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                     unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    
    // Modify query to output JSON format directly
    std::string json_query = query + " FORMAT JSONEachRow";
    
    // Execute query with JSON format
    std::string json_result = query_api_server(json_query);
    
    if (json_result.empty()) {
        strcpy(json_buffer, "[]");
        *length = 2;
        return json_buffer;
    }
    
    // ClickHouse returns JSONEachRow as newline-delimited JSON
    // Convert to JSON array
    std::stringstream result_ss;
    result_ss << "[";
    
    std::stringstream ss(json_result);
    std::string line;
    bool first = true;
    
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        
        if (!first) result_ss << ",";
        first = false;
        
        result_ss << line;
    }
    
    result_ss << "]";
    
    std::string final_json = result_ss.str();
    
    // Copy to buffer
    strncpy(json_buffer, final_json.c_str(), sizeof(json_buffer) - 1);
    json_buffer[sizeof(json_buffer) - 1] = '\0';
    *length = strlen(json_buffer);
    
    return json_buffer;
}

} // extern "C"