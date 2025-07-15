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
#include "common.h"

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
    
public:
    SimpleChDBApiServer() : chdb_handle(nullptr), server_fd(-1), running(false) {}
    
    ~SimpleChDBApiServer() {
        stop();
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool init() {
        // Load chDB library
        const char* lib_paths[] = {
            "/home/cslog/chdb/libchdb.so",
            "./libchdb.so",
            "libchdb.so"
        };
        
        for (const char* path : lib_paths) {
            chdb_handle = dlopen(path, RTLD_LAZY);
            if (chdb_handle) {
                std::cout << "Loaded chdb library from: " << path << std::endl;
                break;
            }
        }
        
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
        
        std::cout << "chDB loaded successfully! (722MB in memory)" << std::endl;
        
        // Warm up
        executeQuery("SELECT 1");
        std::cout << "Server warmed up and ready!" << std::endl;
        
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
        args_storage.push_back("--path=" + CHDB_PATH);
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
        
        std::cout << "Query: " << query << " (time: " << elapsed * 1000 << "ms)" << std::endl;
        
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
            std::cerr << "Socket creation failed" << std::endl;
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
            std::cerr << "Bind failed" << std::endl;
            return false;
        }
        
        if (listen(server_fd, 10) < 0) {
            std::cerr << "Listen failed" << std::endl;
            return false;
        }
        
        running = true;
        std::cout << "\nSimple chDB API Server running on port " << port << std::endl;
        std::cout << "Protocol: Simple binary (no protobuf required)" << std::endl;
        std::cout << "Data path: " << CHDB_PATH << std::endl;
        std::cout << "\nWaiting for connections..." << std::endl;
        
        // Accept loop
        while (running) {
            sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            int client_socket = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            
            if (client_socket < 0) {
                if (running) {
                    std::cerr << "Accept failed" << std::endl;
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
    std::cout << "\nShutting down server..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

int main(int argc, char* argv[]) {
    int port = 8125;
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    g_server = new SimpleChDBApiServer();
    
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