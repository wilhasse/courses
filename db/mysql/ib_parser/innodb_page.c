#include "innodb_page.h"
#include <stdio.h>
#include <string.h>

#define UNIV_PAGE_SIZE 16384

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

void page_header_print(const page_t *page) {
    const byte *bytes = (const byte *)page;
    printf("Page Header Info:\n");
    printf(" Type: %lu\n", page_get_type(page));
    printf(" Page No: %lu\n", page_get_page_no(page));
    printf(" N Records: %lu\n", page_get_n_recs(page));
    printf(" Level: %u\n", bytes[PAGE_HEADER + PAGE_LEVEL] & 0x7F);
    printf(" Index ID: %lu%lu\n",
           mach_read_from_4(bytes + PAGE_HEADER + PAGE_INDEX_ID + 4),
           mach_read_from_4(bytes + PAGE_HEADER + PAGE_INDEX_ID));
}

const byte* get_first_user_rec(const page_t* page) {
    const byte* rec = ((const byte*)page) + PAGE_HEADER;
    /* Find first record using directory */
    ulint n_recs = page_get_n_recs(page);
    if (n_recs == 0) return NULL;
    
    /* Get slot 1 (after infimum) */
    const byte* dir = ((const byte*)page) + PAGE_DIR;
    ulint slot_offset = mach_read_from_2(dir - PAGE_DIR_SLOT_SIZE);
    return ((const byte*)page) + slot_offset;
}

bool is_user_rec(const byte* rec) {
    /* Skip system records (infimum/supremum) */
    if (memcmp(rec, "infimum", 7) == 0 || 
        memcmp(rec, "supremum", 8) == 0) {
        return false;
    }
    return true;
}

ulint get_rec_next_offset(const byte* rec) {
    return mach_read_from_2(rec + REC_NEXT_POS);
}

const byte* get_next_rec(const page_t* page, const byte* rec) {
    ulint next_offset = get_rec_next_offset(rec);
    if (next_offset == 0) return NULL;
    
    const byte* next_rec = ((const byte*)page) + next_offset;
    if (next_rec < (const byte*)page || 
        next_rec >= ((const byte*)page) + UNIV_PAGE_SIZE) {
        return NULL;
    }
    
    return next_rec;
}

bool rec_get_deleted_flag(const byte* rec) {
    return (rec[0] & 0x20) != 0;  /* Bit 5 of first byte */
}

ulint get_rec_field_start(const byte* rec) {
    /* Fields start after record header */
    return REC_HEADER_SIZE;
}
