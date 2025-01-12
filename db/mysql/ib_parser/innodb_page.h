#ifndef INNODB_PAGE_H
#define INNODB_PAGE_H

#include <stdint.h>
#include <stdbool.h>

/* Basic types */
typedef unsigned long ulint;
typedef unsigned char byte;
typedef void* rec_t;
typedef void* page_t;

/* Constants */
#define UNIV_PAGE_SIZE (16 * 1024)
#define PAGE_OLD_INFIMUM  97
#define PAGE_OLD_SUPREMUM 112
#define PAGE_NEW_INFIMUM  97
#define PAGE_NEW_SUPREMUM 112
#define FIL_PAGE_OFFSET 4
#define PAGE_HEADER     38
#define PAGE_N_RECS     16
#define PAGE_LEVEL      26

/* Function declarations */
bool page_is_compact(const page_t *page);
ulint page_get_n_recs(const page_t *page);
ulint page_get_page_no(const page_t *page);
bool page_is_leaf(const page_t *page);
ulint page_offset_get_next(const page_t *page, ulint offset, bool is_comp);
int page_get_deleted_flag(const page_t *page, ulint offset);

#endif /* INNODB_PAGE_H */
