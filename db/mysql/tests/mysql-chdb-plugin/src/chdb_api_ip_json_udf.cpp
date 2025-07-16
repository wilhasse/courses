/**
 * MySQL UDF that connects to chDB API Server with configurable host and returns JSON
 * Usage: chdb_query_json_remote('server:port', 'SELECT ...')
 */

#include <mysql.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
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
#define DEFAULT_PORT 8125
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
        log_file << "[" << timestamp << "] [chdb_api_ip_json_udf] " << message << std::endl;
        log_file.close();
    }
}

extern "C" {
    // Main configurable function - accepts host:port as first parameter
    bool chdb_api_query_json_remote_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    void chdb_api_query_json_remote_deinit(UDF_INIT *initid);
    char* chdb_api_query_json_remote(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                     unsigned long *length, char *is_null, char *error);
    
    // Localhost shortcut function
    bool chdb_api_query_json_local_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    void chdb_api_query_json_local_deinit(UDF_INIT *initid);
    char* chdb_api_query_json_local(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                    unsigned long *length, char *is_null, char *error);
}

// Helper function to check if query already has FORMAT clause
bool hasFormatClause(const std::string& query) {
    std::string upper = query;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    
    size_t pos = upper.rfind("FORMAT");
    if (pos != std::string::npos) {
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
        return query;
    }
    
    std::string trimmed = query;
    size_t end = trimmed.find_last_not_of(" \t\n\r;");
    if (end != std::string::npos) {
        trimmed = trimmed.substr(0, end + 1);
    }
    
    return trimmed + " FORMAT JSON";
}

// Parse host:port string
bool parse_host_port(const std::string& host_port, std::string& host, int& port) {
    size_t colon_pos = host_port.find(':');
    if (colon_pos != std::string::npos) {
        host = host_port.substr(0, colon_pos);
        std::string port_str = host_port.substr(colon_pos + 1);
        port = atoi(port_str.c_str());
        if (port <= 0 || port > 65535) {
            return false;
        }
    } else {
        host = host_port;
        port = DEFAULT_PORT;
    }
    return true;
}

// Simple binary protocol
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

// Connect to server and execute query with JSON format
bool execute_remote_query_json(const std::string& host, int port, const std::string& query,
                              DynamicBuffer& buffer, size_t& response_size, std::string& error_msg) {
    debug_log("execute_remote_query_json called with host=" + host + ", port=" + std::to_string(port) + ", query=" + query);
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        error_msg = "ERROR: Failed to create socket: " + std::string(strerror(errno));
        debug_log(error_msg);
        return false;
    }
    debug_log("Socket created: " + std::to_string(sock));
    
    // Set timeout
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    
    // Resolve hostname
    debug_log("Resolving hostname: " + host);
    struct hostent *server = gethostbyname(host.c_str());
    if (server == NULL) {
        close(sock);
        error_msg = "ERROR: Cannot resolve hostname: " + host + ": " + std::string(hstrerror(h_errno));
        debug_log(error_msg);
        return false;
    }
    debug_log("Hostname resolved successfully");
    
    // Connect to server
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    
    debug_log("Connecting to " + host + ":" + std::to_string(port));
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        error_msg = "ERROR: Cannot connect to " + host + ":" + std::to_string(port) + ": " + std::string(strerror(errno));
        debug_log(error_msg);
        return false;
    }
    debug_log("Connected successfully");
    
    // Send query with JSON format
    std::string json_query = addJsonFormat(query);
    debug_log("Query after adding JSON format: " + json_query);
    
    if (!send_query(sock, json_query)) {
        close(sock);
        error_msg = "ERROR: Failed to send query";
        debug_log(error_msg);
        return false;
    }
    
    // Receive response
    bool success = receive_response(sock, buffer, response_size);
    close(sock);
    
    if (!success) {
        error_msg = response_size > MAX_ALLOWED_SIZE ? 
            "ERROR: Response too large (>1GB)" : "ERROR: Failed to receive response";
        debug_log(error_msg);
        return false;
    }
    
    debug_log("Query completed successfully, received " + format_bytes(response_size));
    return true;
}

// ========== chdb_api_query_json_remote functions ==========

bool chdb_api_query_json_remote_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    debug_log("chdb_api_query_json_remote_init called with " + std::to_string(args->arg_count) + " arguments");
    
    if (args->arg_count != 2) {
        strcpy(message, "chdb_api_query_json_remote() requires exactly two arguments: host:port and SQL query");
        debug_log("Error: Invalid argument count");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT || args->arg_type[1] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_json_remote() requires string arguments");
        return 1;
    }
    
    // Allocate dynamic buffer structure
    DynamicBuffer* buffer = new DynamicBuffer();
    
    initid->ptr = (char*)buffer;
    initid->max_length = MAX_ALLOWED_SIZE;
    initid->maybe_null = 1;
    initid->const_item = 0;
    
    return 0;
}

void chdb_api_query_json_remote_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
        delete buffer;
        initid->ptr = NULL;
    }
}

char* chdb_api_query_json_remote(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1]) {
        debug_log("chdb_api_query_json_remote called with NULL argument(s)");
        *is_null = 1;
        return NULL;
    }
    
    DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
    std::string host_port(args->args[0], args->lengths[0]);
    std::string query(args->args[1], args->lengths[1]);
    debug_log("chdb_api_query_json_remote called with host_port=" + host_port + ", query=" + query);
    
    // Parse host:port
    std::string host;
    int port;
    if (!parse_host_port(host_port, host, port)) {
        const char* err = "ERROR: Invalid host:port format";
        if (buffer->ensure_capacity(strlen(err) + 1)) {
            strcpy(buffer->data, err);
            *length = strlen(err);
            return buffer->data;
        }
        return NULL;
    }
    
    // Execute query with JSON format
    size_t response_size;
    std::string error_msg;
    if (!execute_remote_query_json(host, port, query, *buffer, response_size, error_msg)) {
        if (buffer->ensure_capacity(error_msg.size() + 1)) {
            strcpy(buffer->data, error_msg.c_str());
            *length = error_msg.size();
            return buffer->data;
        }
        return NULL;
    }
    
    *length = response_size;
    return buffer->data;
}

// ========== chdb_api_query_json_local functions ==========

bool chdb_api_query_json_local_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    debug_log("chdb_api_query_json_local_init called with " + std::to_string(args->arg_count) + " arguments");
    
    if (args->arg_count != 1) {
        strcpy(message, "chdb_api_query_json_local() requires exactly one argument: the SQL query");
        debug_log("Error: Invalid argument count");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_json_local() requires a string argument");
        return 1;
    }
    
    // Allocate dynamic buffer structure
    DynamicBuffer* buffer = new DynamicBuffer();
    
    initid->ptr = (char*)buffer;
    initid->max_length = MAX_ALLOWED_SIZE;
    initid->maybe_null = 1;
    initid->const_item = 0;
    
    return 0;
}

void chdb_api_query_json_local_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
        delete buffer;
        initid->ptr = NULL;
    }
}

char* chdb_api_query_json_local(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        debug_log("chdb_api_query_json_local called with NULL argument");
        *is_null = 1;
        return NULL;
    }
    
    DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
    std::string query(args->args[0], args->lengths[0]);
    debug_log("chdb_api_query_json_local called with query=" + query);
    
    // Execute query on localhost:8125 with JSON format
    size_t response_size;
    std::string error_msg;
    if (!execute_remote_query_json("127.0.0.1", DEFAULT_PORT, query, *buffer, response_size, error_msg)) {
        if (buffer->ensure_capacity(error_msg.size() + 1)) {
            strcpy(buffer->data, error_msg.c_str());
            *length = error_msg.size();
            return buffer->data;
        }
        return NULL;
    }
    
    *length = response_size;
    return buffer->data;
}