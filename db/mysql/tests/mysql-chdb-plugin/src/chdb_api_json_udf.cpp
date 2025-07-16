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
#include <fstream>
#include <ctime>
#include <errno.h>

// Configuration
#define CHDB_API_HOST "127.0.0.1"
#define CHDB_API_PORT 8125
#define INITIAL_BUFFER_SIZE 1048576    // 1MB initial buffer
#define MAX_ALLOWED_SIZE 1073741824    // 1GB maximum allowed
#define GROWTH_FACTOR 2                // Double buffer when needed
#define DEBUG_LOG "/tmp/mysql_chdb_api_debug.log"

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

// Helper function to format bytes to human readable string
std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = bytes;
    
    while (size >= 1024 && unit_index < 3) {
        size /= 1024;
        unit_index++;
    }
    
    char buffer[64];
    if (unit_index == 0) {
        snprintf(buffer, sizeof(buffer), "%zu %s", bytes, units[unit_index]);
    } else {
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_index]);
    }
    return std::string(buffer);
}

// Helper function to log debug messages
void debug_log(const std::string& message) {
    std::ofstream log_file(DEBUG_LOG, std::ios::app);
    if (log_file.is_open()) {
        time_t now = time(nullptr);
        char timestamp[100];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
        log_file << "[" << timestamp << "] [chdb_api_json_udf] " << message << std::endl;
        log_file.close();
    }
}

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
    debug_log("Sending query size: " + format_bytes(query.size()));
    
    ssize_t written = write(sock, &size, 4);
    if (written != 4) {
        debug_log("Failed to send query size, wrote " + std::to_string(written) + " bytes: " + std::string(strerror(errno)));
        return false;
    }
    
    written = write(sock, query.c_str(), query.size());
    if (written != (ssize_t)query.size()) {
        debug_log("Failed to send query, wrote " + format_bytes(written) + " of " + format_bytes(query.size()) + ": " + std::string(strerror(errno)));
        return false;
    }
    
    debug_log("Query sent successfully");
    return true;
}

bool receive_response(int sock, DynamicBuffer& buffer, size_t& response_size) {
    uint32_t size;
    debug_log("Reading response size...");
    
    ssize_t bytes_read = read(sock, &size, 4);
    if (bytes_read != 4) {
        debug_log("Failed to read response size, got " + std::to_string(bytes_read) + " bytes: " + std::string(strerror(errno)));
        response_size = 0;
        return false;
    }
    size = ntohl(size);
    response_size = size;
    debug_log("Response size: " + format_bytes(size));
    
    // Ensure buffer can hold the response
    if (!buffer.ensure_capacity(size + 1)) {
        debug_log("Failed to allocate buffer for " + format_bytes(size + 1));
        return false;
    }
    debug_log("Buffer allocated, capacity: " + format_bytes(buffer.capacity));
    
    size_t total_read = 0;
    while (total_read < size) {
        ssize_t n = read(sock, buffer.data + total_read, size - total_read);
        if (n <= 0) {
            debug_log("Failed to read response data at offset " + std::to_string(total_read) + ": " + std::string(strerror(errno)));
            return false;
        }
        total_read += n;
    }
    debug_log("Response received successfully, " + format_bytes(total_read));
    
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
    debug_log("chdb_api_query_json_init called with " + std::to_string(args->arg_count) + " arguments");
    
    if (args->arg_count != 1) {
        strcpy(message, "chdb_api_query_json() requires exactly one argument: the SQL query");
        debug_log("Error: Invalid argument count");
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
        debug_log("chdb_api_query_json called with NULL argument");
        *is_null = 1;
        return NULL;
    }
    
    DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
    std::string query(args->args[0], args->lengths[0]);
    debug_log("chdb_api_query_json called with: " + query);
    
    // Add FORMAT JSON to the query
    query = addJsonFormat(query);
    debug_log("Query after adding JSON format: " + query);
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::string err = "ERROR: Failed to create socket: " + std::string(strerror(errno));
        debug_log(err);
        if (buffer->ensure_capacity(err.size() + 1)) {
            strcpy(buffer->data, err.c_str());
            *length = err.size();
            return buffer->data;
        }
        return NULL;
    }
    debug_log("Socket created: " + std::to_string(sock));
    
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
    
    debug_log("Connecting to " + std::string(CHDB_API_HOST) + ":" + std::to_string(CHDB_API_PORT));
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        char err[256];
        snprintf(err, sizeof(err), "ERROR: Cannot connect to chDB API server at %s:%d: %s", 
                CHDB_API_HOST, CHDB_API_PORT, strerror(errno));
        debug_log(err);
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
    }
    debug_log("Connected successfully");
    
    // Send query
    if (!send_query(sock, query)) {
        close(sock);
        const char* err = "ERROR: Failed to send query";
        debug_log(err);
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
        debug_log(err);
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
    }
    close(sock);
    
    debug_log("Query completed successfully, returning " + format_bytes(response_size));
    *length = response_size;
    return buffer->data;
}