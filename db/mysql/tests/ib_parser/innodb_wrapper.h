#ifndef INNODB_WRAPPER_H
#define INNODB_WRAPPER_H

/* Prevent C++ header issues when compiling as C */
#define HAVE_BOOL 1
typedef char my_bool;

#ifdef __cplusplus
extern "C" {
#endif

/* Basic InnoDB definitions */
typedef unsigned long int ulint;
typedef unsigned char byte;
typedef void* rec_t;
typedef void* page_t;

/* Include minimal required headers */
#include "univ.i"
#include "page0page.h"
#include "rem0rec.h"
#include "mach0data.h"

/* Declare the functions we need */
bool page_is_comp(const page_t *page);
int rec_get_deleted_flag(rec_t *rec, bool comp);
ulint mach_read_from_2(const byte *b);
ulint mach_read_from_4(const byte *b);

#ifdef __cplusplus
}
#endif

#endif /* INNODB_WRAPPER_H */
