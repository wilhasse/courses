#include "innodb_page.h"
#include <stdio.h>
#include <string.h>

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

const byte* get_first_record(const page_t* page) {
    const byte* data = (const byte*)page + PAGE_DATA;
    return data + REC_HEADER_SIZE;  // Skip infimum record
}

const byte* get_next_record(const page_t* page, const byte* rec) {
    ulint next_offset = mach_read_from_2(rec + REC_NEXT_POS);
    if (next_offset == 0 || next_offset >= UNIV_PAGE_SIZE) return NULL;
    return ((const byte*)page) + next_offset;
}

bool is_system_record(const byte* rec) {
    return (rec[REC_N_OWNED] & 0x80) != 0;  // System records have MSB set
}

bool is_deleted_record(const byte* rec) {
    return (rec[REC_STATUS] & REC_DELETED_FLAG) != 0;
}

ulint get_field_offset(const byte* rec, ulint field_no) {
    ulint offset = REC_HEADER_SIZE;  // Start after header
    // Add field lengths for previous fields
    for (ulint i = 0; i < field_no; i++) {
        if (i == 0) offset += 4;  // ID is 4 bytes
        else offset += 100;       // NOME is CHAR(100)
    }
    return offset;
}

ulint get_field_length(const byte* rec, ulint field_no) {
    if (field_no == 0) return 4;  // ID is 4 bytes
    return 100;                   // NOME is CHAR(100)
}
