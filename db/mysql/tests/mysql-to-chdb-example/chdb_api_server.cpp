#include <iostream>
#include <string>
#include <dlfcn.h>
#include <cstring>
#include <vector>
#include <sstream>
#include <thread>
#include <chrono>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <algorithm>
#include "common.h"
#include "chdb_api.pb.h"
#include "logger.h"

// Use the same v2 API structure from query_data_v2.cpp
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

class ChDBApiServer {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    int server_fd;
    bool running;
    std::string chdb_path;
    bool read_only;
    Logger* logger;
    
    // Convert format enum to string
    std::string getFormatString(chdb_api::QueryRequest::OutputFormat format) {
        switch (format) {
            case chdb_api::QueryRequest::CSV: return "CSV";
            case chdb_api::QueryRequest::TSV: return "TabSeparated";
            case chdb_api::QueryRequest::JSON: return "JSON";
            case chdb_api::QueryRequest::PRETTY: return "Pretty";
            case chdb_api::QueryRequest::COMPACT: return "Compact";
            case chdb_api::QueryRequest::VALUES: return "Values";
            default: return "CSV";
        }
    }
    
    // Parse CSV/TSV result into protobuf
    void parseResultToProtobuf(const std::string& data, chdb_api::QueryResponse* response, 
                              chdb_api::QueryRequest::OutputFormat format) {
        if (format == chdb_api::QueryRequest::CSV || format == chdb_api::QueryRequest::TSV) {
            std::istringstream stream(data);
            std::string line;
            char delimiter = (format == chdb_api::QueryRequest::CSV) ? ',' : '\t';
            
            // First line might be headers
            
            while (std::getline(stream, line)) {
                if (line.empty()) continue;
                
                auto* row = response->add_rows();
                std::istringstream line_stream(line);
                std::string value;
                
                while (std::getline(line_stream, value, delimiter)) {
                    auto* val = row->add_values();
                    
                    // Remove quotes if CSV
                    if (format == chdb_api::QueryRequest::CSV && 
                        value.length() >= 2 && value[0] == '"' && value.back() == '"') {
                        value = value.substr(1, value.length() - 2);
                    }
                    
                    // Try to determine type
                    try {
                        if (value == "NULL" || value.empty()) {
                            val->set_is_null(true);
                        } else if (value.find('.') != std::string::npos) {
                            val->set_double_value(std::stod(value));
                        } else {
                            val->set_int_value(std::stoll(value));
                        }
                    } catch (...) {
                        // If not a number, store as string
                        val->set_string_value(value);
                    }
                }
            }
        } else {
            // For other formats, return raw data as a single string value
            auto* row = response->add_rows();
            auto* val = row->add_values();
            val->set_string_value(data);
        }
    }
    
public:
public:
    Logger* getLogger() { return logger; }
    ChDBApiServer(const std::string& path = CHDB_PATH, bool readonly = false) : 
        chdb_handle(nullptr), server_fd(-1), running(false), chdb_path(path), read_only(readonly) {
        logger = new Logger("chdb_api_server.log", true);
    }
    
    ~ChDBApiServer() {
        stop();
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
        if (logger) {
            delete logger;
        }
    }
    
    bool init() {
        // Load chDB library from system path (configured via ldconfig)
        chdb_handle = dlopen("libchdb.so", RTLD_LAZY);
        
        if (!chdb_handle) {
            logger->logError("Failed to load libchdb.so: " + std::string(dlerror()));
            logger->logError("Make sure libchdb.so is installed and ldconfig has been run.");
            return false;
        }
        
        logger->logInfo("Loaded chdb library successfully");
        
        query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
        free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
        
        if (!query_stable_v2 || !free_result_v2) {
            logger->logError("Failed to load functions: " + std::string(dlerror()));
            return false;
        }
        
        logger->logInfo("chDB loaded successfully! (722MB in memory)");
        
        // Warm up
        executeQuery("SELECT 1", chdb_api::QueryRequest::CSV);
        logger->logInfo("Server warmed up and ready!");
        
        return true;
    }
    
    void executeQuery(const std::string& query, chdb_api::QueryRequest::OutputFormat format,
                     chdb_api::QueryResponse* response) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Build arguments
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--output-format=" + getFormatString(format));
        args_storage.push_back("--path=" + chdb_path);
        if (read_only) {
            args_storage.push_back("--readonly=1");  // Read-only mode
        }
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        // Execute query
        struct local_result_v2* result = query_stable_v2(argv.size(), argv.data());
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        
        if (!result) {
            response->set_success(false);
            response->set_error_message("Query execution failed");
            return;
        }
        
        if (result->error_message) {
            response->set_success(false);
            response->set_error_message(result->error_message);
        } else if (result->buf && result->len > 0) {
            response->set_success(true);
            std::string data(result->buf, result->len);
            parseResultToProtobuf(data, response, format);
            response->set_rows_read(result->rows_read);
            response->set_bytes_read(result->bytes_read);
            response->set_elapsed_seconds(elapsed);
        } else {
            response->set_success(true);
            response->set_rows_read(0);
            response->set_elapsed_seconds(elapsed);
        }
        
        free_result_v2(result);
    }
    
    void executeQuery(const std::string& query, chdb_api::QueryRequest::OutputFormat format) {
        chdb_api::QueryResponse response;
        executeQuery(query, format, &response);
        
        if (response.success()) {
            // Calculate result size
            size_t resultBytes = 0;
            for (const auto& row : response.rows()) {
                for (const auto& val : row.values()) {
                    if (val.has_string_value()) {
                        resultBytes += val.string_value().size();
                    } else {
                        resultBytes += 8; // Approximate size for numeric values
                    }
                }
            }
            logger->logQuery(query, response.elapsed_seconds() * 1000, response.rows_size(), 
                           resultBytes, getFormatString(format));
        } else {
            logger->logError("Query failed: " + response.error_message());
        }
    }
    
    void handleClient(int client_socket) {
        // Read request size (4 bytes)
        uint32_t request_size;
        if (read(client_socket, &request_size, 4) != 4) {
            close(client_socket);
            return;
        }
        request_size = ntohl(request_size);
        
        // Read request data
        std::vector<char> buffer(request_size);
        if (read(client_socket, buffer.data(), request_size) != request_size) {
            close(client_socket);
            return;
        }
        
        // Parse protobuf request
        chdb_api::QueryRequest request;
        if (!request.ParseFromArray(buffer.data(), request_size)) {
            logger->logError("Failed to parse request");
            close(client_socket);
            return;
        }
        
        // Log query info - will be replaced with detailed logging after execution
        
        // Execute query
        chdb_api::QueryResponse response;
        executeQuery(request.query(), request.format(), &response);
        
        // Serialize response
        std::string serialized;
        response.SerializeToString(&serialized);
        
        // Log detailed query information
        if (response.success()) {
            size_t resultBytes = serialized.size();
            logger->logQuery(request.query(), response.elapsed_seconds() * 1000, 
                           response.rows_size(), resultBytes, getFormatString(request.format()));
        } else {
            logger->logError("Query failed: " + response.error_message() + " Query: " + request.query());
        }
        
        // Send response size (4 bytes) + response
        uint32_t response_size = htonl(serialized.size());
        write(client_socket, &response_size, 4);
        write(client_socket, serialized.data(), serialized.size());
        
        close(client_socket);
    }
    
    bool start(int port = 8125) {
        // Create socket
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == 0) {
            logger->logError("Socket creation failed");
            return false;
        }
        
        // Allow reuse
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);
        
        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            logger->logError("Bind failed");
            return false;
        }
        
        if (listen(server_fd, 10) < 0) {
            logger->logError("Listen failed");
            return false;
        }
        
        running = true;
        logger->logInfo("\nchDB API Server running on port " + std::to_string(port));
        logger->logInfo("Protocol: Length-prefixed Protocol Buffers");
        logger->logInfo("Data path: " + chdb_path);
        logger->logInfo("Mode: " + std::string(read_only ? "READ-ONLY" : "READ-WRITE"));
        logger->logInfo("\nWaiting for connections...");
        
        // Accept loop
        while (running) {
            sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_socket = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_socket < 0) {
                if (running) {
                    logger->logError("Accept failed");
                }
                continue;
            }
            
            // Handle client in thread
            std::thread client_thread([this, client_socket]() {
                handleClient(client_socket);
            });
            client_thread.detach();
        }
        
        return true;
    }
    
    void stop() {
        running = false;
        if (server_fd >= 0) {
            close(server_fd);
            server_fd = -1;
        }
    }
};

// Global server for signal handling
ChDBApiServer* g_server = nullptr;

void signal_handler(int /*sig*/) {
    if (g_server && g_server->getLogger()) {
        g_server->getLogger()->logInfo("\nShutting down server...");
    } else {
        std::cout << "\nShutting down server..." << std::endl;
    }
    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

int main(int argc, char* argv[]) {
    int port = 8125;
    std::string chdb_path = CHDB_PATH;
    bool read_only = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -p, --port <port>       Port to listen on (default: 8125)" << std::endl;
            std::cout << "  -d, --data <path>       ClickHouse data path (default: " << CHDB_PATH << ")" << std::endl;
            std::cout << "  -r, --readonly          Run in read-only mode (allows concurrent access)" << std::endl;
            std::cout << "  -h, --help              Show this help message" << std::endl;
            return 0;
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--data") && i + 1 < argc) {
            chdb_path = argv[++i];
        } else if (arg == "-r" || arg == "--readonly") {
            read_only = true;
        } else if (std::isdigit(arg[0])) {
            // Backward compatibility: first numeric argument is port
            port = std::atoi(argv[i]);
        }
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    g_server = new ChDBApiServer(chdb_path, read_only);
    
    if (!g_server->init()) {
        std::cerr << "Failed to initialize chDB" << std::endl;
        return 1;
    }
    
    if (!g_server->start(port)) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }
    
    delete g_server;
    return 0;
}