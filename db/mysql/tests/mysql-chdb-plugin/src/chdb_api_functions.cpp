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
#include <fstream>
#include <ctime>
#include <errno.h>

// Configuration
const char* CHDB_API_HOST = "127.0.0.1";
const int CHDB_API_PORT = 8125;
const char* DEBUG_LOG = "/tmp/mysql_chdb_api_debug.log";

extern "C" {

// Helper function to log debug messages
void debug_log(const std::string& message) {
    std::ofstream log_file(DEBUG_LOG, std::ios::app);
    if (log_file.is_open()) {
        time_t now = time(nullptr);
        char timestamp[100];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
        log_file << "[" << timestamp << "] " << message << std::endl;
        log_file.close();
    }
}

// Helper function to connect and query the API server
std::string query_api_server(const std::string& query) {
    debug_log("query_api_server called with: " + query);
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::string error_msg = "ERROR: Failed to create socket: " + std::string(strerror(errno));
        debug_log(error_msg);
        return error_msg;
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
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(CHDB_API_PORT);
    
    if (inet_pton(AF_INET, CHDB_API_HOST, &server_addr.sin_addr) <= 0) {
        std::string error_msg = "ERROR: Invalid address: " + std::string(CHDB_API_HOST);
        debug_log(error_msg);
        close(sock);
        return error_msg;
    }
    
    debug_log("Connecting to " + std::string(CHDB_API_HOST) + ":" + std::to_string(CHDB_API_PORT));
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::string error_msg = "ERROR: Cannot connect to chDB API server: " + std::string(strerror(errno));
        debug_log(error_msg);
        close(sock);
        return error_msg;
    }
    debug_log("Connected successfully");
    
    // Send query size + query
    uint32_t size = htonl(query.size());
    debug_log("Sending query size: " + std::to_string(query.size()));
    
    ssize_t written = write(sock, &size, 4);
    if (written != 4) {
        std::string error_msg = "ERROR: Failed to send query size, wrote " + std::to_string(written) + " bytes: " + std::string(strerror(errno));
        debug_log(error_msg);
        close(sock);
        return error_msg;
    }
    
    written = write(sock, query.c_str(), query.size());
    if (written != (ssize_t)query.size()) {
        std::string error_msg = "ERROR: Failed to send query, wrote " + std::to_string(written) + " of " + std::to_string(query.size()) + " bytes: " + std::string(strerror(errno));
        debug_log(error_msg);
        close(sock);
        return error_msg;
    }
    debug_log("Query sent successfully");
    
    // Read response size
    uint32_t response_size;
    debug_log("Reading response size...");
    
    ssize_t bytes_read = read(sock, &response_size, 4);
    if (bytes_read != 4) {
        std::string error_msg = "ERROR: Failed to read response size, got " + std::to_string(bytes_read) + " bytes: " + std::string(strerror(errno));
        debug_log(error_msg);
        close(sock);
        return error_msg;
    }
    
    response_size = ntohl(response_size);
    debug_log("Response size: " + std::to_string(response_size));
    
    if (response_size == 0) {
        debug_log("WARNING: Response size is 0");
        close(sock);
        return "";
    }
    
    if (response_size > 1048576) { // 1MB sanity check
        std::string error_msg = "ERROR: Response size too large: " + std::to_string(response_size);
        debug_log(error_msg);
        close(sock);
        return error_msg;
    }
    
    // Read response
    std::vector<char> buffer(response_size);
    size_t total_read = 0;
    while (total_read < response_size) {
        ssize_t n = read(sock, buffer.data() + total_read, response_size - total_read);
        if (n <= 0) {
            std::string error_msg = "ERROR: Failed to read response at offset " + std::to_string(total_read) + ": " + std::string(strerror(errno));
            debug_log(error_msg);
            close(sock);
            return error_msg;
        }
        total_read += n;
        debug_log("Read " + std::to_string(n) + " bytes, total: " + std::to_string(total_read) + "/" + std::to_string(response_size));
    }
    
    close(sock);
    
    std::string result(buffer.data(), response_size);
    debug_log("Response received: [" + result + "]");
    
    // Remove trailing newline if present
    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
        debug_log("Removed trailing newline");
    }
    
    debug_log("Returning result: [" + result + "]");
    return result;
}

// UDF: chdb_query(query) - execute any query via API server
static char query_buffer[65535];  // 64KB buffer

bool chdb_query_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    debug_log("chdb_query_init called");
    
    if (args->arg_count != 1) {
        strcpy(message, "chdb_query() requires exactly one argument: the SQL query");
        debug_log("ERROR: Wrong number of arguments");
        return 1;
    }
    
    if (args->arg_type[0] != STRING_RESULT) {
        strcpy(message, "chdb_query() requires a string argument");
        debug_log("ERROR: Argument is not a string");
        return 1;
    }
    
    initid->maybe_null = 1;
    initid->max_length = sizeof(query_buffer) - 1;
    initid->const_item = 0;
    debug_log("chdb_query_init completed successfully");
    return 0;
}

void chdb_query_deinit(UDF_INIT *initid) {
    // Nothing to clean up
}

char *chdb_query(UDF_INIT *initid, UDF_ARGS *args, char *result,
                 unsigned long *length, char *is_null, char *error) {
    debug_log("chdb_query called");
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        debug_log("ERROR: NULL argument");
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    debug_log("Executing query: " + query);
    
    std::string response = query_api_server(query);
    debug_log("Got response: [" + response + "]");
    
    if (response.empty()) {
        debug_log("WARNING: Empty response");
        *is_null = 1;
        return NULL;
    }
    
    if (response.find("ERROR:") == 0) {
        debug_log("Returning error message: " + response);
        strncpy(query_buffer, response.c_str(), sizeof(query_buffer) - 1);
        query_buffer[sizeof(query_buffer) - 1] = '\0';
        *length = strlen(query_buffer);
        return query_buffer;
    }
    
    strncpy(query_buffer, response.c_str(), sizeof(query_buffer) - 1);
    query_buffer[sizeof(query_buffer) - 1] = '\0';
    *length = strlen(query_buffer);
    debug_log("Returning result, length: " + std::to_string(*length));
    return query_buffer;
}

// UDF: chdb_count(table) - get row count from a table
bool chdb_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "chdb_count() requires exactly one argument: table name");
        return 1;
    }
    
    args->arg_type[0] = STRING_RESULT;
    initid->maybe_null = 1;
    return 0;
}

void chdb_count_deinit(UDF_INIT *initid) {}

long long chdb_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return 0;
    }
    
    std::string table(args->args[0], args->lengths[0]);
    std::string query = "SELECT COUNT(*) FROM " + table;
    std::string result = query_api_server(query);
    
    if (result.empty() || result.find("ERROR:") == 0) {
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

// UDF: chdb_sum(table, column) - get sum of a column
bool chdb_sum_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 2) {
        strcpy(message, "chdb_sum() requires exactly two arguments: table name and column name");
        return 1;
    }
    
    args->arg_type[0] = STRING_RESULT;
    args->arg_type[1] = STRING_RESULT;
    initid->maybe_null = 1;
    initid->decimals = 2;
    return 0;
}

void chdb_sum_deinit(UDF_INIT *initid) {}

double chdb_sum(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0] || !args->args[1]) {
        *is_null = 1;
        return 0.0;
    }
    
    std::string table(args->args[0], args->lengths[0]);
    std::string column(args->args[1], args->lengths[1]);
    std::string query = "SELECT SUM(" + column + ") FROM " + table;
    std::string result = query_api_server(query);
    
    if (result.empty() || result.find("ERROR:") == 0) {
        *is_null = 1;
        return 0.0;
    }
    
    try {
        return std::stod(result);
    } catch (...) {
        *is_null = 1;
        return 0.0;
    }
}

} // extern "C"