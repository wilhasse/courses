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
#define INITIAL_BUFFER_SIZE 1048576    // 1MB initial buffer
#define MAX_ALLOWED_SIZE 1073741824    // 1GB maximum allowed
#define GROWTH_FACTOR 2                // Double buffer when needed

// Structure to hold dynamic buffer
struct DynamicBuffer {
    char* data;
    size_t capacity;
    
    DynamicBuffer() : data(nullptr), capacity(0) {}
    
    ~DynamicBuffer() {
        if (data) free(data);
    }
    
    bool ensure_capacity(size_t required) {
        if (required > MAX_ALLOWED_SIZE) return false;
        
        if (required <= capacity) return true;
        
        size_t new_capacity = capacity == 0 ? INITIAL_BUFFER_SIZE : capacity;
        while (new_capacity < required && new_capacity < MAX_ALLOWED_SIZE) {
            new_capacity *= GROWTH_FACTOR;
        }
        if (new_capacity > MAX_ALLOWED_SIZE) new_capacity = MAX_ALLOWED_SIZE;
        
        char* new_data = (char*)realloc(data, new_capacity);
        if (!new_data) return false;
        
        data = new_data;
        capacity = new_capacity;
        return true;
    }
};

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

bool receive_response(int sock, DynamicBuffer& buffer, size_t& response_size) {
    uint32_t size;
    if (read(sock, &size, 4) != 4) {
        response_size = 0;
        return false;
    }
    size = ntohl(size);
    response_size = size;
    
    // Ensure buffer can hold the response
    if (!buffer.ensure_capacity(size + 1)) {
        return false;
    }
    
    size_t total_read = 0;
    while (total_read < size) {
        ssize_t n = read(sock, buffer.data + total_read, size - total_read);
        if (n <= 0) return false;
        total_read += n;
    }
    
    buffer.data[size] = '\0';
    return true;
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
    
    // Allocate dynamic buffer structure
    DynamicBuffer* buffer = new DynamicBuffer();
    
    initid->ptr = (char*)buffer;
    initid->max_length = MAX_ALLOWED_SIZE;
    initid->maybe_null = 1;
    initid->const_item = 0;
    
    return 0;
}

void chdb_api_query_json_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
        delete buffer;
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
    
    DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
    std::string query(args->args[0], args->lengths[0]);
    
    // Add FORMAT JSON to the query
    query = addJsonFormat(query);
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        const char* err = "ERROR: Failed to create socket";
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
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
        char err[256];
        snprintf(err, sizeof(err), "ERROR: Cannot connect to chDB API server at %s:%d", 
                CHDB_API_HOST, CHDB_API_PORT);
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
    }
    
    // Send query
    if (!send_query(sock, query)) {
        close(sock);
        const char* err = "ERROR: Failed to send query";
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
    }
    
    // Receive response
    size_t response_size;
    if (!receive_response(sock, *buffer, response_size)) {
        close(sock);
        const char* err = response_size > MAX_ALLOWED_SIZE ? 
            "ERROR: Response too large (>1GB)" : "ERROR: Failed to receive response";
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
    }
    close(sock);
    
    *length = response_size;
    return buffer->data;
}