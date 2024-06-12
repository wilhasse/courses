#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <poll.h>

#include "common.h"
#include "srvpoll.h"

void fsm_reply_hello(clientstate_t *client, dbproto_hdr_t *hdr) {
    hdr->type = htonl(MSG_HELLO_RESP);
    hdr->len = htons(1);
    dbproto_hello_resp* hello = (dbproto_hello_req*)&hdr[1];
    hello->proto = htons(PROTO_VER);

    write(client->fd, hdr, sizeof(dbproto_hdr_t) + sizeof(dbproto_hello_resp));
}

void fsm_reply_add(clientstate_t *client, dbproto_hdr_t *hdr) {
    hdr->type = htonl(MSG_EMPLOYEE_ADD_RESP);
    hdr->len = htons(0);

    write(client->fd, hdr, sizeof(dbproto_hdr_t));
}

void fsm_reply_hello_err(clientstate_t *client, dbproto_hdr_t *hdr) {
    hdr->type = htonl(MSG_ERROR);
    hdr->len = htons(0);

    write(client->fd, hdr, sizeof(dbproto_hdr_t));
}

void fsm_reply_add_err(clientstate_t *client, dbproto_hdr_t *hdr) {
    hdr->type = htonl(MSG_ERROR);
    hdr->len = htons(0);

    write(client->fd, hdr, sizeof(dbproto_hdr_t));
}

void send_employees(struct dbheader_t *dbhdr, struct employee_t **employeeptr, clientstate_t *client) {

    dbproto_hdr_t *hdr = (dbproto_hdr_t*)client->buffer;
    hdr->type = htonl(MSG_EMPLOYEE_LIST_RESP);
    hdr->len = htons(dbhdr->count);

    write(client->fd, hdr, sizeof(dbproto_hdr_t));

    dbproto_employee_list_resp *employee = (dbproto_hello_req*)&hdr[1];

    struct employee_t *employees = *employeeptr;

    int i = 0;
    for (; i < dbhdr->count; i++) {
        strncpy(&employee->name, employees[i].name, sizeof(employee->name));
        strncpy(&employee->address, employees[i].address, sizeof(employee->address));
        employee->hours = htonl(employees[i].hours);
        write(client->fd, employee, sizeof(dbproto_employee_list_resp));
    }
}

void handle_client_fsm(struct dbheader_t *dbhdr, struct employee_t **employeeptr, clientstate_t *client, int dbfd) {
    dbproto_hdr_t *hdr = (dbproto_hdr_t*)client->buffer; 

    hdr->type = ntohl(hdr->type);
    hdr->len = ntohs(hdr->len);

    if (client->state == STATE_HELLO) {
        if (hdr->type != MSG_HELLO_REQ || hdr->len != 1) {
            printf("Didn't get MSG_HELLO in HELLO state...\n");
            fsm_reply_hello_err(client, hdr);
            return;
        }

        dbproto_hello_req* hello = (dbproto_hello_req*)&hdr[1];
        hello->proto = ntohs(hello->proto);
        if (hello->proto != PROTO_VER) {
            printf("Protocol mismatch...\n");
            fsm_reply_hello_err(client, hdr);
            return;
        }

        fsm_reply_hello(client, hdr);
        client->state = STATE_MSG;
        printf("Client upgraded to STATE_MSG\n");
    }

    if (client->state == STATE_MSG) {
        if (hdr->type == MSG_EMPLOYEE_ADD_REQ) {

            dbproto_employee_add_req* employee = (dbproto_hello_req*)&hdr[1];

            printf("Adding employee: %s\n", employee->data);
            if (add_employee(dbhdr, employeeptr, employee->data) != STATUS_SUCCESS) {
                fsm_reply_add_err(client, hdr);
                return;
            } else {
                fsm_reply_add(client,hdr);
                output_file(dbfd, dbhdr, *employeeptr);
            }
        }

        if (hdr->type == MSG_EMPLOYEE_LIST_REQ) {
            printf("Listing employees\n");
            send_employees(dbhdr, employeeptr, client);
        }
    }
}

void init_clients(clientstate_t* states) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        states[i].fd = -1; // -1 indicates a free slot
        states[i].state = STATE_NEW;
        memset(&states[i].buffer, '\0', BUFF_SIZE);
    }
}

int find_free_slot(clientstate_t* states) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (states[i].fd == -1) {
            return i;
        }
    }
    return -1; // No free slot found
}

int find_slot_by_fd(clientstate_t* states, int fd) {
    for (int i = 0; i < MAX_CLIENTS; i++) {
        if (states[i].fd == fd) {
            return i;
        }
    }
    return -1; // Not found
}