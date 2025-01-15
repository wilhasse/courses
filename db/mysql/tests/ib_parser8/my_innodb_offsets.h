#pragma once
#include <cstdint>   // for uint, etc.
#include "page0page.h" // for rec_t, etc.
#include "rem0rec.h"   // for REC_OFFS_SQL_NULL, REC_OFFS_MASK, etc.

// Example: replicate old 5.7 logic for getting the size of nth field
inline unsigned long my_rec_offs_nth_size(const unsigned long* offsets, unsigned long nth)
{
    unsigned long start = offsets[nth];
    unsigned long end   = offsets[nth + 1];

    // If flagged as SQL NULL, return UNIV_SQL_NULL
    if (end & REC_OFFS_SQL_NULL) {
        return UNIV_SQL_NULL;
    }
    start &= REC_OFFS_MASK;
    end   &= REC_OFFS_MASK;

    return (end > start) ? (end - start) : 0;
}

inline bool my_rec_offs_nth_extern(const unsigned long* offsets, unsigned long nth)
{
    // In 5.7, external fields had REC_OFFS_EXTERNAL. If you want that logic:
    unsigned long end = offsets[nth + 1];
    return (end & REC_OFFS_EXTERNAL) != 0;
}

// A minimal “get field pointer” that simply adds the offset to rec
inline unsigned char* my_rec_get_nth_field(
    unsigned char* rec,       // rec_t* pointer
    const unsigned long* offs,
    unsigned long nth_field,
    unsigned long* length_out)
{
    unsigned long start = offs[nth_field];
    unsigned long end   = offs[nth_field + 1];

    if (end & REC_OFFS_SQL_NULL) {
        if (length_out) {
            *length_out = UNIV_SQL_NULL;
        }
        return nullptr;
    }
    start &= REC_OFFS_MASK;
    end   &= REC_OFFS_MASK;

    unsigned long field_len = (end > start) ? (end - start) : 0;
    if (length_out) {
        *length_out = field_len;
    }
    return rec + start; // pointer arithmetic
}

// If you also call something like 'my_rec_get_nth_field_size(rec, i)':
inline unsigned long my_rec_get_nth_field_size(
    unsigned char* rec, // if you really need rec
    unsigned long nth)
{
    // Or just do the same logic as my_rec_offs_nth_size(),
    // if your offsets array is global or stored somewhere accessible
    // For simplicity, you might remove references to "my_rec_get_nth_field_size()" 
    // if you don't actually need it.
    return 0;  // stub
}

