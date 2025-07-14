#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    std::cout << "Simple TCP connection test to chDB API server" << std::endl;
    
    // Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket creation failed: " << strerror(errno) << std::endl;
        return 1;
    }
    
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8125);
    
    // Try different addresses
    const char* addresses[] = {"127.0.0.1", "localhost", "0.0.0.0"};
    
    for (const char* addr : addresses) {
        std::cout << "\nTrying to connect to " << addr << ":8125..." << std::endl;
        
        if (strcmp(addr, "localhost") == 0) {
            // For localhost, use loopback IP
            inet_pton(AF_INET, "127.0.0.1", &server_addr.sin_addr);
        } else {
            if (inet_pton(AF_INET, addr, &server_addr.sin_addr) <= 0) {
                std::cerr << "Invalid address format for " << addr << std::endl;
                continue;
            }
        }
        
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) >= 0) {
            std::cout << "SUCCESS: Connected to server at " << addr << ":8125" << std::endl;
            close(sock);
            return 0;
        } else {
            std::cerr << "Failed to connect: " << strerror(errno) << std::endl;
        }
        
        // Create new socket for next attempt
        close(sock);
        sock = socket(AF_INET, SOCK_STREAM, 0);
    }
    
    close(sock);
    std::cerr << "\nERROR: Could not connect to server on any address" << std::endl;
    std::cerr << "Make sure chdb_api_server is running" << std::endl;
    return 1;
}