#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>

int main() {

	struct sockaddr_in serverAddress = {0};
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_addr.s_addr = 0;
	serverAddress.sin_port = htons(5555);

	struct sockaddr_in clientAddress;
	socklen_t clientAddrLen = sizeof(clientAddress);

	int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (serverSocket == -1) {
		perror("socket");
		return 0;
	}

	printf("Our socket FD is %d\n", serverSocket);

	if (bind(serverSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1) {
	    perror("bind");
	    return 0;
	}

	if (listen(serverSocket, 0) == -1) {
	    perror("listen");
	    return 0;
	}

	while (1) {

		int clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddress, &clientAddrLen);

		if (clientSocket == -1) {
		    perror("accept");
		    return 0;
		}

		//handle_client(clientSocket);

		close(clientSocket);

	}
}
