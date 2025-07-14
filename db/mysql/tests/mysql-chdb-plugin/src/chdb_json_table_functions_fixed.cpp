/**
 * Fixed version of MySQL UDF functions that return JSON for use with JSON_TABLE
 * Addresses character set issues in MySQL 8.0.19+
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

// ========== Simple JSON Function ==========

// chdb_customers_json() - Get all customers as JSON using ClickHouse's JSON format
bool chdb_customers_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 0) {
        strcpy(message, "chdb_customers_json() takes no arguments");
        return 1;
    }
    
    initid->maybe_null = 1;
    initid->max_length = sizeof(json_buffer) - 1;
    initid->const_item = 0;
    return 0;
}

void chdb_customers_json_deinit(UDF_INIT *initid) {}

char* chdb_customers_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                         unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    // Use ClickHouse's native JSON format
    std::string query = "SELECT id, name, email, age, city FROM mysql_import.customers ORDER BY id FORMAT JSON";
    std::string json_result = query_api_server(query);
    
    if (json_result.empty()) {
        strcpy(json_buffer, "{\"data\":[]}");
        *length = strlen(json_buffer);
        return json_buffer;
    }
    
    // ClickHouse JSON format includes metadata, extract just the data array
    // Find the "data" array in the response
    size_t data_pos = json_result.find("\"data\":");
    if (data_pos != std::string::npos) {
        size_t array_start = json_result.find("[", data_pos);
        size_t array_end = json_result.rfind("]");
        if (array_start != std::string::npos && array_end != std::string::npos) {
            std::string data_array = json_result.substr(array_start, array_end - array_start + 1);
            strncpy(json_buffer, data_array.c_str(), sizeof(json_buffer) - 1);
            json_buffer[sizeof(json_buffer) - 1] = '\0';
            *length = strlen(json_buffer);
            return json_buffer;
        }
    }
    
    // Fallback: return the full response
    strncpy(json_buffer, json_result.c_str(), sizeof(json_buffer) - 1);
    json_buffer[sizeof(json_buffer) - 1] = '\0';
    *length = strlen(json_buffer);
    
    return json_buffer;
}

// ========== Alternative: Return raw string for CAST ==========

// chdb_customers_json_raw() - Return raw result for manual CAST
bool chdb_customers_json_raw_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 0) {
        strcpy(message, "chdb_customers_json_raw() takes no arguments");
        return 1;
    }
    
    initid->maybe_null = 1;
    initid->max_length = sizeof(json_buffer) - 1;
    initid->const_item = 0;
    return 0;
}

void chdb_customers_json_raw_deinit(UDF_INIT *initid) {}

char* chdb_customers_json_raw(UDF_INIT *initid, UDF_ARGS *args, char *result,
                             unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    // Simple JSON array construction from TSV
    std::string query = "SELECT id, name, email, age, city FROM mysql_import.customers ORDER BY id";
    std::string tsv_result = query_api_server(query);
    
    if (tsv_result.empty()) {
        strcpy(json_buffer, "[]");
        *length = 2;
        return json_buffer;
    }
    
    // Convert TSV to simple JSON array
    std::stringstream json;
    json << "[";
    
    std::stringstream ss(tsv_result);
    std::string line;
    bool first_row = true;
    
    while (std::getline(ss, line)) {
        if (line.empty()) continue;
        
        if (!first_row) json << ",";
        first_row = false;
        
        // Parse TSV line
        std::vector<std::string> fields;
        std::stringstream line_ss(line);
        std::string field;
        while (std::getline(line_ss, field, '\t')) {
            fields.push_back(field);
        }
        
        if (fields.size() >= 5) {
            json << "{";
            json << "\"id\":" << fields[0] << ",";
            json << "\"name\":\"" << fields[1] << "\",";
            json << "\"email\":\"" << fields[2] << "\",";
            json << "\"age\":" << fields[3] << ",";
            json << "\"city\":\"" << fields[4] << "\"";
            json << "}";
        }
    }
    
    json << "]";
    
    std::string final_json = json.str();
    strncpy(json_buffer, final_json.c_str(), sizeof(json_buffer) - 1);
    json_buffer[sizeof(json_buffer) - 1] = '\0';
    *length = strlen(json_buffer);
    
    return json_buffer;
}

// ========== Debug function ==========

// chdb_test_json() - Simple test function
bool chdb_test_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    initid->maybe_null = 0;
    initid->max_length = 1000;
    initid->const_item = 1;
    return 0;
}

void chdb_test_json_deinit(UDF_INIT *initid) {}

char* chdb_test_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                    unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    const char* test_json = "[{\"id\":1,\"name\":\"Test User\",\"age\":25}]";
    strcpy(json_buffer, test_json);
    *length = strlen(json_buffer);
    
    return json_buffer;
}

} // extern "C"