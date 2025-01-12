#ifndef INNODB_PAGE_H
#define INNODB_PAGE_H

#include <stdint.h>
#include <stdbool.h>

/* Basic types */
typedef unsigned long ulint;
typedef unsigned char byte;
typedef void* rec_t;
typedef void* page_t;

/* File page types */
#define FIL_PAGE_INDEX          17855
#define FIL_PAGE_TYPE_FSP_HDR   8
#define FIL_PAGE_IBUF_BITMAP    5
#define FIL_PAGE_TYPE_ALLOCATED 0
#define FIL_PAGE_INODE         3
#define FIL_PAGE_TYPE_SYS      6

/* Page header offsets */
#define FIL_PAGE_SPACE_OR_CHKSUM 0
#define FIL_PAGE_OFFSET          4
#define FIL_PAGE_PREV            8
#define FIL_PAGE_NEXT           12
#define FIL_PAGE_LSN            16
#define FIL_PAGE_TYPE           24
#define FIL_PAGE_FILE_FLUSH_LSN 26
#define FIL_PAGE_ARCH_LOG_NO    34

/* Page directory offsets */
#define PAGE_HEADER             38
#define PAGE_N_DIR_SLOTS        0
#define PAGE_HEAP_TOP           2
#define PAGE_N_HEAP             4
#define PAGE_FREE               6
#define PAGE_GARBAGE            8
#define PAGE_LAST_INSERT       10
#define PAGE_DIRECTION         12
#define PAGE_N_DIRECTION       14
#define PAGE_N_RECS            16
#define PAGE_MAX_TRX_ID        18
#define PAGE_LEVEL             26
#define PAGE_INDEX_ID          28
#define PAGE_BTR_SEG_LEAF      36

/* Record related constants */
#define PAGE_OLD_INFIMUM       99
#define PAGE_OLD_SUPREMUM      112
#define REC_HEADER_SIZE        6
#define REC_NEXT_POS           2
#define REC_INFO_BITS_SHIFT    0
#define REC_INFO_DELETED_MASK  0x20

/* Function declarations */
bool page_is_index(const page_t *page);
ulint page_get_type(const page_t *page);
bool page_is_leaf(const page_t *page);
ulint page_get_n_recs(const page_t *page);
ulint page_get_page_no(const page_t *page);
const byte* page_get_infimum_rec(const page_t* page);
const byte* page_get_supremum_rec(const page_t* page);
const byte* page_rec_get_next(const byte* rec, const page_t* page);
bool rec_get_deleted_flag(const byte* rec);
ulint mach_read_from_2(const byte *b);
ulint mach_read_from_4(const byte *b);
void page_header_print(const page_t *page);

#endif /* INNODB_PAGE_H */
