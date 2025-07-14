#include <mysql.h>
#include <curl/curl.h>
#include <cstring>
#include <string>
#include <sstream>

// Configuration
const char* CHDB_SERVER_URL = "http://localhost:8123/";

extern "C" {

// Callback for CURL to capture response
size_t write_callback(char* ptr, size_t size, size_t nmemb, std::string* data) {
    data->append(ptr, size * nmemb);
    return size * nmemb;
}

// Helper function to execute query via HTTP
std::string execute_http_query(const std::string& query) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "";
    }
    
    std::string response;
    
    // Create JSON request
    std::string json_request = "{\"query\": \"" + query + "\"}";
    
    // Set up CURL
    curl_easy_setopt(curl, CURLOPT_URL, CHDB_SERVER_URL);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_request.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    
    // Headers
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Timeout
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    // Execute
    CURLcode res = curl_easy_perform(curl);
    
    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        return "";
    }
    
    // Extract result from JSON response
    // Simple extraction - look for "data": or "result":
    size_t data_pos = response.find("\"data\":");
    if (data_pos != std::string::npos) {
        size_t start = response.find("[", data_pos);
        size_t end = response.find("]", start);
        if (start != std::string::npos && end != std::string::npos) {
            return response.substr(start, end - start + 1);
        }
    }
    
    // For scalar results
    size_t result_pos = response.find("\"result\":");
    if (result_pos != std::string::npos) {
        size_t start = response.find("\"", result_pos + 9);
        size_t end = response.find("\"", start + 1);
        if (start != std::string::npos && end != std::string::npos) {
            return response.substr(start + 1, end - start - 1);
        }
    }
    
    return response;
}

// UDF: ch_query_http() - execute query via HTTP server
static char result_buffer[65536];

bool ch_query_http_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_query_http() requires exactly one argument");
        return 1;
    }
    
    args->arg_type[0] = STRING_RESULT;
    initid->maybe_null = 1;
    initid->max_length = sizeof(result_buffer) - 1;
    
    // Initialize CURL globally
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    return 0;
}

void ch_query_http_deinit(UDF_INIT *initid) {
    curl_global_cleanup();
}

char* ch_query_http(UDF_INIT *initid, UDF_ARGS *args, char *result,
                   unsigned long *length, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return NULL;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    std::string response = execute_http_query(query);
    
    if (response.empty()) {
        *is_null = 1;
        return NULL;
    }
    
    strncpy(result_buffer, response.c_str(), sizeof(result_buffer) - 1);
    result_buffer[sizeof(result_buffer) - 1] = '\0';
    *length = strlen(result_buffer);
    
    return result_buffer;
}

// UDF: ch_query_http_scalar() - for simple scalar results
bool ch_query_http_scalar_init(UDF_INIT *initid, UDF_ARGS *args, char *message) {
    if (args->arg_count != 1) {
        strcpy(message, "ch_query_http_scalar() requires exactly one argument");
        return 1;
    }
    
    args->arg_type[0] = STRING_RESULT;
    initid->maybe_null = 1;
    initid->max_length = 1024;
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    return 0;
}

void ch_query_http_scalar_deinit(UDF_INIT *initid) {
    curl_global_cleanup();
}

long long ch_query_http_scalar(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error) {
    *is_null = 0;
    *error = 0;
    
    if (!args->args[0]) {
        *is_null = 1;
        return 0;
    }
    
    std::string query(args->args[0], args->lengths[0]);
    std::string response = execute_http_query(query);
    
    if (response.empty()) {
        *is_null = 1;
        return 0;
    }
    
    try {
        return std::stoll(response);
    } catch (...) {
        *is_null = 1;
        return 0;
    }
}

} // extern "C"