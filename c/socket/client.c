#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

	if (argc != 2) {
		printf("Usage: %s <ip of the host>\n", argv[0]);
		return 0;
	}

	struct sockaddr_in serverAddress = {0};
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_addr.s_addr = inet_addr(argv[1]);
	serverAddress.sin_port = htons(5555);

	int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
	if (clientSocket == -1) {
		perror("socket");
		return 0;
	}

	printf("Our socket FD is %d\n", clientSocket);

	if (connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1) {
	    perror("connect");
	    close(clientSocket);
	    return 0;
	}
}
