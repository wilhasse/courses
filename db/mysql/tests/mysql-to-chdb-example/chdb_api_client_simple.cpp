#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

void showHelp(const char* program) {
    std::cout << "Usage: " << program << " [options] \"SQL query\"" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  -h, --host <host>    Server host (default: 127.0.0.1)" << std::endl;
    std::cout << "  -p, --port <port>    Server port (default: 8125)" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << program << " \"SELECT COUNT(*) FROM mysql_import.historico\"" << std::endl;
    std::cout << "  " << program << " -h 192.168.1.10 \"SELECT * FROM mysql_import.historico LIMIT 5\"" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        showHelp(argv[0]);
        return 1;
    }
    
    std::string query;
    std::string host = "127.0.0.1";
    int port = 8125;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") {
            showHelp(argv[0]);
            return 0;
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg[0] != '-') {
            query = arg;
            break;
        }
    }
    
    if (query.empty()) {
        std::cerr << "Error: No query provided" << std::endl;
        showHelp(argv[0]);
        return 1;
    }
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return 1;
    }
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Invalid address: " << host << std::endl;
        close(sock);
        return 1;
    }
    
    std::cout << "Connecting to " << host << ":" << port << "..." << std::endl;
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Connection failed" << std::endl;
        close(sock);
        return 1;
    }
    
    // Send query size (4 bytes) + query
    uint32_t query_size = htonl(query.size());
    if (write(sock, &query_size, 4) != 4) {
        std::cerr << "Failed to send query size" << std::endl;
        close(sock);
        return 1;
    }
    
    if (write(sock, query.c_str(), query.size()) != query.size()) {
        std::cerr << "Failed to send query" << std::endl;
        close(sock);
        return 1;
    }
    
    // Read response size (4 bytes)
    uint32_t response_size;
    if (read(sock, &response_size, 4) != 4) {
        std::cerr << "Failed to read response size" << std::endl;
        close(sock);
        return 1;
    }
    response_size = ntohl(response_size);
    
    // Read response
    std::vector<char> buffer(response_size);
    size_t total_read = 0;
    while (total_read < response_size) {
        ssize_t n = read(sock, buffer.data() + total_read, response_size - total_read);
        if (n <= 0) {
            std::cerr << "Failed to read response" << std::endl;
            close(sock);
            return 1;
        }
        total_read += n;
    }
    
    std::string response(buffer.data(), response_size);
    std::cout << "Response:\n" << response << std::endl;
    
    close(sock);
    return 0;
}