#ifndef INNODB_PAGE_H
#define INNODB_PAGE_H

#include <stdint.h>
#include <stdbool.h>

/* Basic types */
typedef unsigned long ulint;
typedef unsigned char byte;
typedef void* rec_t;
typedef void* page_t;

/* Constants from MySQL 5.7 */
#define UNIV_PAGE_SIZE (16 * 1024)
#define FIL_PAGE_DATA 38
#define PAGE_HEADER 38
#define PAGE_N_RECS 16
#define PAGE_LEVEL 26
#define FIL_PAGE_OFFSET 4
#define PAGE_NEW_INFIMUM  97
#define PAGE_NEW_SUPREMUM 112
#define PAGE_OLD_INFIMUM  99
#define PAGE_OLD_SUPREMUM 112
#define REC_N_NEW_EXTRA_BYTES 5
#define REC_N_OLD_EXTRA_BYTES 6

/* Record field offsets */
#define REC_NEXT        2  /* offset of next record pointer */
#define REC_OLD_SHORT   3  /* offset of short data in old-style record */
#define REC_OLD_N_FIELDS 4 /* offset of n_fields in old-style record */

/* Function declarations */
bool page_is_compact(const page_t *page);
ulint page_get_n_recs(const page_t *page);
ulint page_get_page_no(const page_t *page);
bool page_is_leaf(const page_t *page);
ulint page_get_infimum_offset(bool is_compact);
ulint page_get_supremum_offset(bool is_compact);
ulint page_get_next_offset(const page_t *page, ulint current_offset, bool is_compact);
ulint mach_read_from_2(const byte *b);
ulint mach_read_from_4(const byte *b);

#endif /* INNODB_PAGE_H */
