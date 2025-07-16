/**
 * Simplified chDB API Server for MySQL UDF
 * Uses simple binary protocol instead of Protocol Buffers
 */

#include <iostream>
#include <string>
#include <dlfcn.h>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <algorithm>
#include "common.h"
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

class SimpleChDBApiServer {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    int server_fd;
    bool running;
    std::string chdb_path;
    bool read_only;
    Logger* logger;
    
public:
    Logger* getLogger() { return logger; }
    SimpleChDBApiServer(const std::string& path = CHDB_PATH, bool readonly = false) : 
        chdb_handle(nullptr), server_fd(-1), running(false), chdb_path(path), read_only(readonly) {
        logger = new Logger("chdb_api_server_simple.log", true);
    }
    
    ~SimpleChDBApiServer() {
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
        executeQuery("SELECT 1");
        logger->logInfo("Server warmed up and ready!");
        
        return true;
    }
    
    std::string executeQuery(const std::string& query) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Build arguments
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--output-format=TabSeparated");  // TSV format for MySQL
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
        
        std::string response;
        
        if (!result) {
            response = "ERROR: Query execution failed";
        } else if (result->error_message) {
            response = std::string("ERROR: ") + result->error_message;
        } else if (result->buf && result->len > 0) {
            response = std::string(result->buf, result->len);
            // Remove trailing newline if present
            if (!response.empty() && response.back() == '\n') {
                response.pop_back();
            }
        } else {
            response = ""; // Empty result
        }
        
        if (result) {
            free_result_v2(result);
        }
        
        // Calculate result size and row count
        size_t resultBytes = response.size();
        size_t resultRows = 0;
        if (result && result->rows_read > 0) {
            resultRows = result->rows_read;
        } else if (!response.empty() && response.find("ERROR") == std::string::npos) {
            // Count rows for TSV format
            resultRows = std::count(response.begin(), response.end(), '\n');
            if (!response.empty() && response.back() != '\n') {
                resultRows++; // Count last row if not ending with newline
            }
        }
        
        logger->logQuery(query, elapsed * 1000, resultRows, resultBytes, "TSV");
        
        return response;
    }
    
    void handleClient(int client_socket) {
        // Read query size (4 bytes)
        uint32_t query_size;
        if (read(client_socket, &query_size, 4) != 4) {
            close(client_socket);
            return;
        }
        query_size = ntohl(query_size);
        
        // Sanity check
        if (query_size > 1048576) { // 1MB max query
            close(client_socket);
            return;
        }
        
        // Read query
        std::vector<char> buffer(query_size);
        size_t total_read = 0;
        while (total_read < query_size) {
            ssize_t n = read(client_socket, buffer.data() + total_read, query_size - total_read);
            if (n <= 0) {
                close(client_socket);
                return;
            }
            total_read += n;
        }
        
        std::string query(buffer.data(), query_size);
        
        // Execute query
        std::string result = executeQuery(query);
        
        // Send response size (4 bytes) + response
        uint32_t response_size = htonl(result.size());
        write(client_socket, &response_size, 4);
        write(client_socket, result.data(), result.size());
        
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
        logger->logInfo("\nSimple chDB API Server running on port " + std::to_string(port));
        logger->logInfo("Protocol: Simple binary (no protobuf required)");
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
SimpleChDBApiServer* g_server = nullptr;

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
    
    g_server = new SimpleChDBApiServer(chdb_path, read_only);
    
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