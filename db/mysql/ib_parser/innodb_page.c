#include "innodb_page.h"
#include <stdio.h>

ulint mach_read_from_2(const byte *b) {
    return ((ulint)(b[0]) << 8) | (ulint)(b[1]);
}

ulint mach_read_from_4(const byte *b) {
    return ((ulint)(b[0]) << 24) |
           ((ulint)(b[1]) << 16) |
           ((ulint)(b[2]) << 8)  |
            (ulint)(b[3]);
}

bool page_is_index(const page_t *page) {
    return page_get_type(page) == FIL_PAGE_INDEX;
}

ulint page_get_type(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return mach_read_from_2(bytes + FIL_PAGE_TYPE);
}

bool page_is_leaf(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return (bytes[PAGE_HEADER + PAGE_LEVEL] & 0x7F) == 0;
}

ulint page_get_n_recs(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return mach_read_from_2(bytes + PAGE_HEADER + PAGE_N_RECS);
}

ulint page_get_page_no(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return mach_read_from_4(bytes + FIL_PAGE_OFFSET);
}

const byte* page_get_infimum_rec(const page_t* page) {
    return ((const byte*)page) + PAGE_OLD_INFIMUM;
}

const byte* page_get_supremum_rec(const page_t* page) {
    return ((const byte*)page) + PAGE_OLD_SUPREMUM;
}

bool rec_get_deleted_flag(const byte* rec) {
    return (rec[0] & REC_INFO_DELETED_MASK) != 0;
}

const byte* page_rec_get_next(const byte* rec, const page_t* page) {
    ulint next = mach_read_from_2(rec + REC_NEXT_POS);
    if (next == 0) return NULL;
    return ((const byte*)page) + next;
}

void page_header_print(const page_t *page) {
    const byte *bytes = (const byte *)page;
    
    printf("Page Header Info:\n");
    printf("  Type: %lu\n", page_get_type(page));
    printf("  Page No: %lu\n", page_get_page_no(page));
    printf("  Prev Page: %lu\n", mach_read_from_4(bytes + FIL_PAGE_PREV));
    printf("  Next Page: %lu\n", mach_read_from_4(bytes + FIL_PAGE_NEXT));
    printf("  N Records: %lu\n", page_get_n_recs(page));
    printf("  Level: %u\n", bytes[PAGE_HEADER + PAGE_LEVEL] & 0x7F);
    printf("  Index ID: %lu%lu\n", 
           mach_read_from_4(bytes + PAGE_HEADER + PAGE_INDEX_ID + 4),
           mach_read_from_4(bytes + PAGE_HEADER + PAGE_INDEX_ID));
}
