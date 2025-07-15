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

// Configuration
#define DEFAULT_PORT 8125
#define MAX_RESULT_SIZE 10485760  // 10MB max result

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

// Connect to server and execute query
std::string execute_remote_query(const std::string& host, int port, const std::string& query) {
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return "ERROR: Failed to create socket";
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
        return "ERROR: Cannot resolve hostname: " + host;
    }
    
    // Connect to server
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    memcpy(&server_addr.sin_addr.s_addr, server->h_addr, server->h_length);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(sock);
        return "ERROR: Cannot connect to " + host + ":" + std::to_string(port);
    }
    
    // Send query
    if (!send_query(sock, query)) {
        close(sock);
        return "ERROR: Failed to send query";
    }
    
    // Receive response
    std::string response = receive_response(sock);
    close(sock);
    
    return response.empty() ? "ERROR: Empty response from server" : response;
}

// ========== chdb_api_query_json_remote functions ==========

bool chdb_api_query_json_remote_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 2) {
        strcpy(message, "chdb_api_query_json_remote() requires exactly two arguments: host:port and SQL query");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT || args->arg_type[1] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_json_remote() requires string arguments");
        return 1;
    }
    
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

void chdb_api_query_json_remote_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        free(initid->ptr);
        initid->ptr = NULL;
    }
}

char* chdb_api_query_json_remote(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string host_port(args->args[0], args->lengths[0]);
    std::string query(args->args[1], args->lengths[1]);
    
    // Add FORMAT JSON
    query = addJsonFormat(query);
    
    // Parse host:port
    std::string host;
    int port;
    if (!parse_host_port(host_port, host, port)) {
        strcpy(initid->ptr, "ERROR: Invalid host:port format");
        *length = strlen(initid->ptr);
        return initid->ptr;
    }
    
    // Execute query
    std::string response = execute_remote_query(host, port, query);
    
    // Copy response to buffer
    size_t copy_len = std::min(response.size(), (size_t)MAX_RESULT_SIZE - 1);
    memcpy(initid->ptr, response.c_str(), copy_len);
    initid->ptr[copy_len] = '\0';
    *length = copy_len;
    
    return initid->ptr;
}

// ========== chdb_api_query_json_local functions ==========

bool chdb_api_query_json_local_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_api_query_json_local() requires exactly one argument: the SQL query");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT) {
        strcpy(message, "chdb_api_query_json_local() requires a string argument");
        return 1;
    }
    
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

void chdb_api_query_json_local_deinit(UDF_INIT *initid) {
    if (initid->ptr) {
        free(initid->ptr);
        initid->ptr = NULL;
    }
}

char* chdb_api_query_json_local(UDF_INIT *initid, UDF_ARGS *args, char *result,
                                unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    
    // Add FORMAT JSON
    query = addJsonFormat(query);
    
    // Execute query on localhost:8125
    std::string response = execute_remote_query("127.0.0.1", DEFAULT_PORT, query);
    
    // Copy response to buffer
    size_t copy_len = std::min(response.size(), (size_t)MAX_RESULT_SIZE - 1);
    memcpy(initid->ptr, response.c_str(), copy_len);
    initid->ptr[copy_len] = '\0';
    *length = copy_len;
    
    return initid->ptr;
}