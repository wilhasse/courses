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

bool page_is_compact(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return (bytes[PAGE_HEADER + PAGE_LEVEL] & 0x80) != 0;
}

ulint page_get_n_recs(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return mach_read_from_2(bytes + PAGE_HEADER + PAGE_N_RECS);
}

ulint page_get_page_no(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return mach_read_from_4(bytes + FIL_PAGE_OFFSET);
}

bool page_is_leaf(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return (bytes[PAGE_HEADER + PAGE_LEVEL] & 0x7F) == 0;
}

ulint page_get_infimum_offset(bool is_compact) {
    return is_compact ? PAGE_NEW_INFIMUM : PAGE_OLD_INFIMUM;
}

ulint page_get_supremum_offset(bool is_compact) {
    return is_compact ? PAGE_NEW_SUPREMUM : PAGE_OLD_SUPREMUM;
}

ulint page_get_next_offset(const page_t *page, ulint current_offset, bool is_compact) {
    const byte *bytes = (const byte *)page;
    
    /* Get the pointer to the next record */
    ulint next_ptr = mach_read_from_2(bytes + current_offset - REC_NEXT);
    
    /* In compact format, next_ptr is relative to current position */
    if (is_compact) {
        return current_offset + next_ptr;
    } else {
        return next_ptr;
    }
}
