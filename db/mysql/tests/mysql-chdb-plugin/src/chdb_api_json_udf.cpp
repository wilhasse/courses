/**
 * MySQL UDF that connects to chDB API Server and returns JSON format
 * This is a variant of chdb_api_udf that automatically appends FORMAT JSON
 */

#include <mysql.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

// Configuration
#define CHDB_API_HOST "127.0.0.1"
#define CHDB_API_PORT 8125
#define MAX_RESULT_SIZE 10485760  // 10MB max result

extern "C" {
    bool chdb_api_query_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    void chdb_api_query_json_deinit(UDF_INIT *initid);
    char* chdb_api_query_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                              unsigned long *length, char *is_null, char *error);
}

// Simple binary protocol without protobuf dependency
// Format: [4 bytes size][query string]
// Response: [4 bytes size][result string]

bool send_query(int sock, const std::string& query) {
    uint32_t size = htonl(query.size());
    if (write(sock, &size, 4) != 4) return false;
    if (write(sock, query.c_str(), query.size()) != (ssize_t)query.size()) return false;
    return true;
}

std::string receive_response(int sock) {
    uint32_t size;
    if (read(sock, &size, 4) != 4) return "";
    size = ntohl(size);
    
    if (size > MAX_RESULT_SIZE) return "ERROR: Response too large";
    
    std::vector<char> buffer(size);
    size_t total_read = 0;
    while (total_read < size) {
        ssize_t n = read(sock, buffer.data() + total_read, size - total_read);
        if (n <= 0) return "ERROR: Failed to read response";
        total_read += n;
    }
    
    return std::string(buffer.data(), size);
}

// Helper function to check if query already has FORMAT clause
bool hasFormatClause(const std::string& query) {
    std::string upper = query;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    // Look for FORMAT keyword
    size_t pos = upper.rfind("FORMAT");
    if (pos != std::string::npos) {
        // Check if it's actually a FORMAT clause (not part of a string or identifier)
        size_t semicolon = upper.find(';', pos);
        if (semicolon == std::string::npos || semicolon > pos) {
            return true;
        }
    }
    return false;
}

// Helper function to add FORMAT JSON to query
std::string addJsonFormat(const std::string& query) {
    if (hasFormatClause(query)) {
        return query;  // Already has FORMAT clause
    }
    
    // Find the end of the query (before any trailing semicolon)
    std::string trimmed = query;
    size_t end = trimmed.find_last_not_of(" \t\n\r;");
    if (end != std::string::npos) {
        trimmed = trimmed.substr(0, end + 1);
    }
    
    return trimmed + " FORMAT JSON";
}

bool chdb_api_query_json_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_api_query_json() requires exactly one argument: the SQL query");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_json() requires a string argument");
        return 1;
    }
    
    // Force string argument to be treated as UTF-8
    args->arg_type[0] = STRING_RESULT;
    
    // Allocate buffer for results
    initid->ptr = (char*)malloc(MAX_RESULT_SIZE);
    if (!initid->ptr) {
        strcpy(message, "Failed to allocate memory");
        return 1;
    }
    
    initid->max_length = MAX_RESULT_SIZE;
    initid->maybe_null = 1;
    initid->const_item = 0;
    
    return 0;
}

void chdb_api_query_json_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        free(initid->ptr);
        initid->ptr = NULL;
    }
}

char* chdb_api_query_json(UDF_INIT *initid, UDF_ARGS *args, char *result,
                          unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    
    // Add FORMAT JSON to the query
    query = addJsonFormat(query);
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        strcpy(initid->ptr, "ERROR: Failed to create socket");
        *length = strlen(initid->ptr);
        return initid->ptr;
    }
    
    // Set timeout
    struct timeval tv;
    tv.tv_sec = 30;  // 30 second timeout
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    
    // Connect to server
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(CHDB_API_PORT);
    inet_pton(AF_INET, CHDB_API_HOST, &server_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        snprintf(initid->ptr, MAX_RESULT_SIZE, 
                "ERROR: Cannot connect to chDB API server at %s:%d", 
                CHDB_API_HOST, CHDB_API_PORT);
        *length = strlen(initid->ptr);
        return initid->ptr;
    }
    
    // Send query
    if (!send_query(sock, query)) {
        close(sock);
        strcpy(initid->ptr, "ERROR: Failed to send query");
        *length = strlen(initid->ptr);
        return initid->ptr;
    }
    
    // Receive response
    std::string response = receive_response(sock);
    close(sock);
    
    if (response.empty()) {
        strcpy(initid->ptr, "ERROR: Empty response from server");
        *length = strlen(initid->ptr);
        return initid->ptr;
    }
    
    // Copy response to buffer
    size_t copy_len = std::min(response.size(), (size_t)MAX_RESULT_SIZE - 1);
    memcpy(initid->ptr, response.c_str(), copy_len);
    initid->ptr[copy_len] = '\0';
    *length = copy_len;
    
    return initid->ptr;
}