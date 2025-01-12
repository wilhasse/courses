#define UNIV_MUST_NOT_INLINE
#define UNIV_INLINE
#define HAVE_BOOL 1

#include "innodb_page.h"

/* Helper functions */
static ulint mach_read_from_2(const byte *b) {
    return ((ulint)(b[0]) << 8) | (ulint)(b[1]);
}

static ulint mach_read_from_4(const byte *b) {
    return ((ulint)(b[0]) << 24) |
           ((ulint)(b[1]) << 16) |
           ((ulint)(b[2]) << 8)  |
            (ulint)(b[3]);
}

bool page_is_compact(const page_t *page) {
    const byte *bytes = (const byte *)page;
    return (mach_read_from_2(bytes + 26) & 0x8000) != 0;
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
    return mach_read_from_2(bytes + PAGE_HEADER + PAGE_LEVEL) == 0;
}

ulint page_offset_get_next(const page_t *page, ulint offset, bool is_comp) {
    const byte *bytes = (const byte *)page;
    if (is_comp) {
        ulint next_offset = mach_read_from_2(bytes + offset - 2);
        return offset + next_offset;
    } else {
        return mach_read_from_2(bytes + offset - 2);
    }
}

int page_get_deleted_flag(const page_t *page, ulint offset) {
    const byte *bytes = (const byte *)page;
    return (bytes[offset - 6] >> 5) & 1;  // Simplified deletion flag check
}
