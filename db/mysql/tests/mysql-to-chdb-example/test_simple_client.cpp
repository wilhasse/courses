#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " \"SQL_QUERY\"" << std::endl;
        return 1;
    }
    
    std::string query = argv[1];
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }
    
    // Connect
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8125);
    inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect: " << strerror(errno) << std::endl;
        close(sock);
        return 1;
    }
    
    std::cout << "Connected to server" << std::endl;
    
    // Send query size + query
    uint32_t size = htonl(query.size());
    if (write(sock, &size, 4) != 4) {
        std::cerr << "Failed to send size" << std::endl;
        close(sock);
        return 1;
    }
    
    if (write(sock, query.c_str(), query.size()) != (ssize_t)query.size()) {
        std::cerr << "Failed to send query" << std::endl;
        close(sock);
        return 1;
    }
    
    std::cout << "Query sent: " << query << std::endl;
    
    // Read response size
    uint32_t response_size;
    ssize_t n = read(sock, &response_size, 4);
    if (n != 4) {
        std::cerr << "Failed to read response size, got " << n << " bytes" << std::endl;
        close(sock);
        return 1;
    }
    
    response_size = ntohl(response_size);
    std::cout << "Response size: " << response_size << std::endl;
    
    // Read response
    std::vector<char> buffer(response_size);
    size_t total = 0;
    while (total < response_size) {
        n = read(sock, buffer.data() + total, response_size - total);
        if (n <= 0) {
            std::cerr << "Failed to read response" << std::endl;
            close(sock);
            return 1;
        }
        total += n;
    }
    
    close(sock);
    
    std::string result(buffer.data(), response_size);
    std::cout << "Result: [" << result << "]" << std::endl;
    
    return 0;
}