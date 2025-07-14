#include <iostream>
#include <string>
#include <dlfcn.h>
#include <vector>
#include <sstream>
#include <cstring>
#include <microhttpd.h>
#include <json/json.h>

// chDB structures
struct local_result_v2 {
    char * buf;
    size_t len;
    void * _vec;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char * error_message;
};

typedef struct local_result_v2* (*query_stable_v2_fn)(int argc, char** argv);
typedef void (*free_result_v2_fn)(struct local_result_v2* result);

class ChDBServer {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    std::string data_path;
    
public:
    ChDBServer(const std::string& path) : chdb_handle(nullptr), data_path(path) {}
    
    ~ChDBServer() {
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool init() {
        std::cout << "Loading libchdb.so..." << std::endl;
        chdb_handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
        if (!chdb_handle) {
            std::cerr << "Failed to load libchdb.so: " << dlerror() << std::endl;
            return false;
        }
        
        query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
        free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
        
        if (!query_stable_v2 || !free_result_v2) {
            std::cerr << "Failed to load functions: " << dlerror() << std::endl;
            return false;
        }
        
        std::cout << "chDB loaded successfully!" << std::endl;
        return true;
    }
    
    std::string execute_query(const std::string& query) {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--output-format=JSON");
        args_storage.push_back("--path=" + data_path);
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        struct local_result_v2* result = query_stable_v2(argv.size(), argv.data());
        
        if (!result) {
            return "{\"error\": \"Query execution failed\"}";
        }
        
        std::string output;
        if (result->error_message) {
            Json::Value error;
            error["error"] = result->error_message;
            Json::FastWriter writer;
            output = writer.write(error);
        } else if (result->buf && result->len > 0) {
            output = std::string(result->buf, result->len);
        } else {
            output = "{\"data\": [], \"rows\": 0}";
        }
        
        free_result_v2(result);
        return output;
    }
};

// Global server instance
ChDBServer* g_server = nullptr;

// HTTP request handler
static int handle_request(void* cls,
                         struct MHD_Connection* connection,
                         const char* url,
                         const char* method,
                         const char* version,
                         const char* upload_data,
                         size_t* upload_data_size,
                         void** con_cls) {
    
    static int dummy;
    if (&dummy != *con_cls) {
        *con_cls = &dummy;
        return MHD_YES;
    }
    
    if (strcmp(method, "POST") != 0) {
        return MHD_NO;
    }
    
    if (*upload_data_size != 0) {
        std::string* data = (std::string*)*con_cls;
        if (!data) {
            data = new std::string();
            *con_cls = data;
        }
        data->append(upload_data, *upload_data_size);
        *upload_data_size = 0;
        return MHD_YES;
    }
    
    std::string* request_data = (std::string*)*con_cls;
    if (!request_data) {
        return MHD_NO;
    }
    
    // Parse JSON request
    Json::Reader reader;
    Json::Value request;
    std::string response_str;
    
    if (reader.parse(*request_data, request)) {
        std::string query = request.get("query", "").asString();
        if (!query.empty()) {
            response_str = g_server->execute_query(query);
        } else {
            response_str = "{\"error\": \"No query provided\"}";
        }
    } else {
        response_str = "{\"error\": \"Invalid JSON\"}";
    }
    
    // Send response
    struct MHD_Response* response = MHD_create_response_from_buffer(
        response_str.size(),
        (void*)response_str.c_str(),
        MHD_RESPMEM_MUST_COPY
    );
    
    MHD_add_response_header(response, "Content-Type", "application/json");
    int ret = MHD_queue_response(connection, MHD_HTTP_OK, response);
    MHD_destroy_response(response);
    
    delete request_data;
    *con_cls = nullptr;
    
    return ret;
}

int main(int argc, char* argv[]) {
    int port = 8123;
    std::string data_path = "/home/cslog/courses/db/mysql/tests/mysql-to-chdb-example/clickhouse_data";
    
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    if (argc > 2) {
        data_path = argv[2];
    }
    
    std::cout << "Starting chDB HTTP Server..." << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    
    g_server = new ChDBServer(data_path);
    if (!g_server->init()) {
        std::cerr << "Failed to initialize chDB" << std::endl;
        return 1;
    }
    
    struct MHD_Daemon* daemon = MHD_start_daemon(
        MHD_USE_SELECT_INTERNALLY,
        port,
        NULL, NULL,
        &handle_request, NULL,
        MHD_OPTION_END
    );
    
    if (!daemon) {
        std::cerr << "Failed to start HTTP server" << std::endl;
        return 1;
    }
    
    std::cout << "Server running on http://localhost:" << port << std::endl;
    std::cout << "POST JSON to / with format: {\"query\": \"SELECT ...\"}" << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    
    // Keep running
    while (true) {
        sleep(1);
    }
    
    MHD_stop_daemon(daemon);
    delete g_server;
    
    return 0;
}