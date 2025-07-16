/**
 * MySQL UDF that connects to chDB API Server with configurable host
 * Usage: chdb_query_remote('server:port', 'SELECT ...')
 *        chdb_query_remote('192.168.1.100:8125', 'SELECT ...')
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

// Configuration
#define DEFAULT_PORT 8125
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
    // Main configurable function - accepts host:port as first parameter
    bool chdb_api_query_remote_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    void chdb_api_query_remote_deinit(UDF_INIT *initid);
    char* chdb_api_query_remote(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                unsigned long *length, char *is_null, char *error);
    
    // Localhost shortcut function - uses 127.0.0.1:8125
    bool chdb_api_query_local_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
    void chdb_api_query_local_deinit(UDF_INIT *initid);
    char* chdb_api_query_local(UDF_INIT *initid, UDF_ARGS *args, char *result,
                               unsigned long *length, char *is_null, char *error);
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

// Connect to server and execute query
bool execute_remote_query(const std::string& host, int port, const std::string& query, 
                         DynamicBuffer& buffer, size_t& response_size, std::string& error_msg) {
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        error_msg = "ERROR: Failed to create socket";
        return false;
    }
    
    // Set timeout
    struct timeval tv;
    tv.tv_sec = 30;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    
    // Resolve hostname
    struct hostent *server = gethostbyname(host.c_str());
    if (server == NULL) {
        close(sock);
        error_msg = "ERROR: Cannot resolve hostname: " + host;
        return false;
    }
    
    // Connect to server
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        error_msg = "ERROR: Cannot connect to " + host + ":" + std::to_string(port);
        return false;
    }
    
    // Send query
    if (!send_query(sock, query)) {
        close(sock);
        error_msg = "ERROR: Failed to send query";
        return false;
    }
    
    // Receive response
    bool success = receive_response(sock, buffer, response_size);
    close(sock);
    
    if (!success) {
        error_msg = response_size > MAX_ALLOWED_SIZE ? 
            "ERROR: Response too large (>1GB)" : "ERROR: Failed to receive response";
        return false;
    }
    
    return true;
}

// ========== chdb_api_query_remote functions ==========

bool chdb_api_query_remote_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 2) {
        strcpy(message, "chdb_api_query_remote() requires exactly two arguments: host:port and SQL query");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT || args->arg_type[1] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_remote() requires string arguments");
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

void chdb_api_query_remote_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
        delete buffer;
        initid->ptr = NULL;
    }
}

char* chdb_api_query_remote(UDF_INIT *initid, UDF_ARGS *args, char *result,
                            unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1]) {
        *is_null = 1;
        return NULL;
    }
    
    DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
    std::string host_port(args->args[0], args->lengths[0]);
    std::string query(args->args[1], args->lengths[1]);
    
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
    
    // Execute query
    size_t response_size;
    std::string error_msg;
    if (!execute_remote_query(host, port, query, *buffer, response_size, error_msg)) {
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

// ========== chdb_api_query_local functions (localhost shortcut) ==========

bool chdb_api_query_local_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_api_query_local() requires exactly one argument: the SQL query");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_local() requires a string argument");
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

void chdb_api_query_local_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
        delete buffer;
        initid->ptr = NULL;
    }
}

char* chdb_api_query_local(UDF_INIT *initid, UDF_ARGS *args, char *result,
                           unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return NULL;
    }
    
    DynamicBuffer* buffer = (DynamicBuffer*)initid->ptr;
    std::string query(args->args[0], args->lengths[0]);
    
    // Execute query on localhost:8125
    size_t response_size;
    std::string error_msg;
    if (!execute_remote_query("127.0.0.1", DEFAULT_PORT, query, *buffer, response_size, error_msg)) {
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