#ifndef PROTO_HDR_H
#define PROTO_HDR_H

typedef enum {
    PROTO_HELLO,
} proto_type_e;

typedef struct {
    proto_type_e type;
    unsigned int len;
} proto_hdr_t;

#endif // PROTO_HDR_H
