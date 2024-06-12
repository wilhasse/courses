#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>

#include "common.h"
#include "file.h"
#include "parse.h"
#include "srvpoll.h"

#define MAX_CLIENTS 256

clientstate_t clientStates[MAX_CLIENTS] = {0};

void print_usage(char *argv[]) {
	printf("Usage: %s -n -f <database file>\n", argv[0]);
	printf("\t -n  - create new database file\n");
	printf("\t -f  - (required) path to database file\n");
	printf("\t -p  - (required) port to listen to\n");
	return;
}

void poll_loop(unsigned short port, struct dbheader_t *dbhdr, struct employee_t **employeeptr, int dbfd) {
	int listen_fd, conn_fd, freeSlot;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    struct pollfd fds[MAX_CLIENTS + 1];
    int nfds = 1; 
    int opt = 1;

    // Initialize client states
    init_clients(&clientStates);

    // Create listening socket
    if ((listen_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    // Set up server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // Bind
    if (bind(listen_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    // Listen
    if (listen(listen_fd, 10) == -1) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    printf("Server listening on port %d\n", PORT);

    memset(fds, 0, sizeof(fds));
    fds[0].fd = listen_fd;
    fds[0].events = POLLIN;
    nfds = 1;

    while (1) {

        int ii = 1;
        for (int i = 0; i < MAX_CLIENTS; i++) {
            if (clientStates[i].fd != -1) {
                fds[ii].fd = clientStates[i].fd; // Offset by 1 for listen_fd
                fds[ii].events = POLLIN;
                ii++;
            }   
        }

        // Wait for an event on one of the sockets
        int n_events = poll(fds, nfds, -1); // -1 means no timeout
        if (n_events == -1) {
            perror("poll");
            exit(EXIT_FAILURE);
        }

        // Check for new connections
        if (fds[0].revents & POLLIN) {
            if ((conn_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &client_len)) == -1) {
                perror("accept");
                continue;
            }

            printf("New connection from %s:%d\n",
            	inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

            freeSlot = find_free_slot(&clientStates);
            if (freeSlot == -1) {
                printf("Server full: closing new connection\n");
                close(conn_fd);
            } else {
                clientStates[freeSlot].fd = conn_fd;
                clientStates[freeSlot].state = STATE_HELLO;
                nfds++;
                printf("Slot %d has fd %d\n", freeSlot, clientStates[freeSlot].fd);
            }

            n_events--;
        }

        // Check each client for read/write activity
        for (int i = 1; i <= nfds && n_events > 0; i++) { // Start from 1 to skip the listen_fd
            if (fds[i].revents & POLLIN) {
                n_events--;

                int fd = fds[i].fd;
                int slot = find_slot_by_fd(&clientStates, fd);
                ssize_t bytes_read = read(fd, &clientStates[slot].buffer, sizeof(clientStates[slot].buffer));
                if (bytes_read <= 0) {
                    // Connection closed or error
                    close(fd);

                    if (slot != -1) {
                        clientStates[slot].fd = -1; // Free up the slot
                        clientStates[slot].state = STATE_DISCONNECTED;
                        printf("Client disconnected\n");
                        nfds--;
                    }
                } else {
                    handle_client_fsm(dbhdr, employeeptr, &clientStates[slot], dbfd);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) { 
	char *filepath = NULL;
	char *portarg = NULL;
	unsigned short port = 0;
	bool newfile = false;
	bool list = false;
	int c;

	int dbfd = -1;
	struct dbheader_t *dbhdr = NULL;
	struct employee_t *employees = NULL;

	while ((c = getopt(argc, argv, "nf:p:")) != -1) {
		switch (c) {
			case 'n':
				newfile = true;
				break;
			case 'f':
				filepath = optarg;
				break;
			case 'p':
				portarg = optarg;
				port = atoi(portarg);
				if (port == 0) {
					printf("Bad port: %s\n", portarg);
				}
				break;
			case '?':
				printf("Unknown option -%c\n", c);
			default:
				return -1;

		}
	}

	if (filepath == NULL) {
		printf("Filepath is a required argument\n");
		print_usage(argv);

		return 0;
	}

	if (port == 0) {
		printf("Port not set\n");
		print_usage(argv);
		return 0;
	}


	if (newfile) {
		dbfd = create_db_file(filepath);
		if (dbfd == STATUS_ERROR) {
			printf("Unable to create database file\n");
			return -1;
		}

		if (create_db_header(dbfd, &dbhdr) == STATUS_ERROR) {
			printf("Failed to create database header\n"); 
			return -1;
		}

	} else {
		dbfd = open_db_file(filepath);
		if (dbfd == STATUS_ERROR) {
			printf("Unable to open database file\n");
			return -1;
		}

		if (validate_db_header(dbfd, &dbhdr) == STATUS_ERROR) {
			printf("Failed to validate database header\n");
			return -1;
		}
	}

	if (read_employees(dbfd, dbhdr, &employees) != STATUS_SUCCESS) {
		printf("Failed to read employees");
		return 0;
	}

	poll_loop(port, dbhdr, &employees, dbfd);

	return 0;
}
