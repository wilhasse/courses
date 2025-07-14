#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <errno.h>
#include "chdb_api.pb.h"

class ChDBApiClient {
private:
    std::string server_host;
    int server_port;
    
public:
    ChDBApiClient(const std::string& host = "127.0.0.1", int port = 8125) 
        : server_host(host), server_port(port) {}
    
    bool executeQuery(const std::string& query, 
                     chdb_api::QueryRequest::OutputFormat format,
                     chdb_api::QueryResponse& response) {
        // Create socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Socket creation failed" << std::endl;
            return false;
        }
        
        struct sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(server_port);
        
        if (inet_pton(AF_INET, server_host.c_str(), &server_addr.sin_addr) <= 0) {
            std::cerr << "Invalid address: " << server_host << std::endl;
            std::cerr << "Error: " << strerror(errno) << std::endl;
            close(sock);
            return false;
        }
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Connection failed to " << server_host << ":" << server_port << std::endl;
            std::cerr << "Error: " << strerror(errno) << std::endl;
            close(sock);
            return false;
        }
        
        // Create request
        chdb_api::QueryRequest request;
        request.set_query(query);
        request.set_format(format);
        
        // Serialize request
        std::string serialized;
        request.SerializeToString(&serialized);
        
        // Send size + request
        uint32_t size = htonl(serialized.size());
        write(sock, &size, 4);
        write(sock, serialized.data(), serialized.size());
        
        // Read response size
        uint32_t response_size;
        if (read(sock, &response_size, 4) != 4) {
            close(sock);
            return false;
        }
        response_size = ntohl(response_size);
        
        // Read response
        std::vector<char> buffer(response_size);
        if (read(sock, buffer.data(), response_size) != response_size) {
            close(sock);
            return false;
        }
        
        close(sock);
        
        // Parse response
        return response.ParseFromArray(buffer.data(), response_size);
    }
    
    void printResponse(const chdb_api::QueryResponse& response) {
        if (!response.success()) {
            std::cout << "Error: " << response.error_message() << std::endl;
            return;
        }
        
        std::cout << "Success! Rows: " << response.rows_size() 
                  << ", Time: " << response.elapsed_seconds() * 1000 << "ms"
                  << ", Rows read: " << response.rows_read() << std::endl;
        
        // Print data
        for (const auto& row : response.rows()) {
            for (int i = 0; i < row.values_size(); i++) {
                if (i > 0) std::cout << "\t";
                
                const auto& val = row.values(i);
                if (val.is_null()) {
                    std::cout << "NULL";
                } else if (val.has_int_value()) {
                    std::cout << val.int_value();
                } else if (val.has_double_value()) {
                    std::cout << val.double_value();
                } else if (val.has_string_value()) {
                    std::cout << val.string_value();
                } else if (val.has_bool_value()) {
                    std::cout << (val.bool_value() ? "true" : "false");
                }
            }
            std::cout << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " \"SQL_QUERY\" [format] [host] [port]" << std::endl;
        std::cout << "Formats: CSV, TSV, JSON, PRETTY, COMPACT, VALUES" << std::endl;
        std::cout << "Example: " << argv[0] << " \"SELECT COUNT(*) FROM mysql_import.customers\"" << std::endl;
        std::cout << "Example: " << argv[0] << " \"SELECT 1\" CSV 127.0.0.1 8125" << std::endl;
        return 1;
    }
    
    std::string query = argv[1];
    chdb_api::QueryRequest::OutputFormat format = chdb_api::QueryRequest::CSV;
    std::string host = "127.0.0.1";
    int port = 8125;
    
    if (argc > 2) {
        std::string format_str = argv[2];
        if (format_str == "TSV") format = chdb_api::QueryRequest::TSV;
        else if (format_str == "JSON") format = chdb_api::QueryRequest::JSON;
        else if (format_str == "PRETTY") format = chdb_api::QueryRequest::PRETTY;
        else if (format_str == "COMPACT") format = chdb_api::QueryRequest::COMPACT;
        else if (format_str == "VALUES") format = chdb_api::QueryRequest::VALUES;
    }
    
    if (argc > 3) {
        host = argv[3];
    }
    
    if (argc > 4) {
        port = std::atoi(argv[4]);
    }
    
    std::cout << "Connecting to " << host << ":" << port << "..." << std::endl;
    ChDBApiClient client(host, port);
    chdb_api::QueryResponse response;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!client.executeQuery(query, format, response)) {
        std::cerr << "Failed to execute query" << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Total round-trip time: " << total_time << "ms" << std::endl;
    std::cout << "Server processing time: " << response.elapsed_seconds() * 1000 << "ms" << std::endl;
    std::cout << "Network + parsing overhead: " << total_time - (response.elapsed_seconds() * 1000) << "ms" << std::endl;
    std::cout << std::endl;
    
    client.printResponse(response);
    
    return 0;
}