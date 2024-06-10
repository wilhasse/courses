#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include "protocol.h"

void handle_server(int fd) {
	char buf[4096] = {0};
	proto_hdr_t *hdr = (proto_hdr_t*)buf;
	read(fd, hdr, sizeof(proto_hdr_t) + sizeof(int));
	hdr->type = ntohl(hdr->type); // unpack the type
	hdr->len = ntohs(hdr->len);

	int *data = &hdr[1];
	*data = ntohl(*data); // protocol version one, packed

	if (*data != 1) {
		printf("Protocol mismatch!\n");
		return;
	}

	printf("Successfully connected to the server, protocol v1\n");
	return;
}

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
        handle_server(clientSocket);
}
