#ifndef INNODB_PAGE_H
#define INNODB_PAGE_H

#include <stdint.h>
#include <stdbool.h>

/* Basic types */
typedef unsigned long ulint;
typedef unsigned char byte;
typedef void* rec_t;
typedef void* page_t;

/* Page types */
#define FIL_PAGE_INDEX          17855
#define FIL_PAGE_TYPE_FSP_HDR   8
#define FIL_PAGE_IBUF_BITMAP    5
#define FIL_PAGE_TYPE_ALLOCATED 0
#define FIL_PAGE_INODE         3
#define FIL_PAGE_TYPE_SYS      6

/* Page header offsets */
#define FIL_PAGE_OFFSET          4
#define FIL_PAGE_TYPE           24
#define PAGE_HEADER             38
#define PAGE_N_RECS            16
#define PAGE_LEVEL             26
#define PAGE_INDEX_ID          28
#define PAGE_HEAP_TOP          0
#define PAGE_N_HEAP            2
#define PAGE_FREE              4
#define PAGE_DATA              38

/* Record constants */
#define REC_NEXT_POS            2
#define REC_HEADER_SIZE         6
#define REC_N_OWNED            3
#define REC_HEAP_NO            4
#define REC_STATUS             5
#define REC_DELETED_FLAG       0x20

/* Record format offsets */
#define REC_OLD_N_FIELDS       4
#define REC_OLD_NEXT_POS       2
#define REC_OLD_SHORT_FIELD    3

#define UNIV_PAGE_SIZE 16384

/* Function declarations */
bool page_is_index(const page_t *page);
ulint page_get_type(const page_t *page);
bool page_is_leaf(const page_t *page);
ulint page_get_n_recs(const page_t *page);
ulint page_get_page_no(const page_t *page);
ulint mach_read_from_2(const byte *b);
ulint mach_read_from_4(const byte *b);
void page_header_print(const page_t *page);

/* Record navigation */
const byte* get_first_record(const page_t* page);
const byte* get_next_record(const page_t* page, const byte* rec);
bool is_system_record(const byte* rec);
bool is_deleted_record(const byte* rec);
ulint get_field_offset(const byte* rec, ulint field_no);
ulint get_field_length(const byte* rec, ulint field_no);

#endif /* INNODB_PAGE_H */
