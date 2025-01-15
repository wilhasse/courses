# Concepts

Below is a more **in-depth explanation** of the “old style” and “new style” (compact) record layouts, and how InnoDB uses “canonical coordinates” to represent records. We will also walk through the examples given in the **canonical coordinate** comments (like `("AA", SQL-NULL, "BB", "")`), showing how they relate to InnoDB’s internal record format.

---

## 1. Old-Style Physical Record Layout

### Structure Overview

```
| offset of last field + bits ... |
| offset of first field + bits ... |
| 4 bits: delete-mark + min-record |
| 4 bits: number of records owned  |
| 13 bits: heap number             |
| 10 bits: number of fields        |
| 1 bit: if offsets are 1-byte (1) or 2-byte (0) |
| 2 bytes: absolute pointer to next record in page |
ORIGIN (start of fields)
| First field data    |
...
| Last field data     |
```

1. **Two bytes pointing to the next record**: This is often called the “next record pointer” on the page.
2. **1 bit for offset-size**:  
   - If this bit = 1, all field offsets are 1 byte each.  
   - If this bit = 0, all field offsets are 2 bytes each.  
   - The offset values themselves (in the array above the data) are stored in **reverse order** relative to the start of the data.
3. **10 bits for number of fields**: Tells how many columns (fields) this record has.
4. **13 bits for heap number**: Identifies the record’s position in that page’s “row heap” (the internal structure used to place records).
5. **4 bits for # of records owned** + **4 bits for delete-mark**:  
   - “Delete mark” is a single bit (plus some extra bits used for infimum/supremum or min-record indicators).  
   - “# of records owned” is how many subsequent “physically consecutive” records in the page are “owned” by this record—a detail of InnoDB’s older page organization logic.
6. **Offsets array**: For each field, we store “the offset from the origin to the **end** of that field.”  
   - Each offset might have high bits used for `SQL-NULL` or `extern` flags:  
     - If it’s 2-byte offsets, the highest bit indicates `NULL`, and the second-highest bit indicates “externally stored” (e.g., a big BLOB).  
     - If it’s 1-byte offsets, only the highest bit is used for `NULL`. “Extern” cannot be expressed in 1-byte offsets.

### Example of “offset array” usage

Suppose you have a record with 3 fields: `("ABC", "XYZ", NULL)` in old-style format, using **2-byte** offsets:

1. Let’s say the origin is right before `"ABC"`.  
2. The total length is:  
   - `"ABC"` = 3 bytes,  
   - `"XYZ"` = 3 bytes,  
   - `NULL` = 0 bytes physically, but we store an offset with the `SQL-NULL` bit set.  
3. We store offsets (in reverse order) above the data:
   - Field 3 (NULL): End offset is `6 + 0 = 6`, or we keep it the same as the previous one, **plus** we set the “NULL” bit in that offset.  
   - Field 2 (“XYZ”): The end offset is `3 (for ABC) + 3 (for XYZ) = 6`.  
   - Field 1 (“ABC”): The end offset is `3`.
4. Then we put them in **reverse** order in the record header structure.  

In short, if you peek into the page, you find something like (offset array) → (record header bits) → (data, “ABCXYZ”).

---

## 2. New-Style (Compact) Physical Record Layout

### Structure Overview

```
| length byte(s) of last var-length field | 
...
| length byte(s) of first var-length field |
| (null flags, bit-per-nullable-col) |
| 4 bits: delete-mark + min-record |
| 4 bits: number of records owned        |
| 13 bits: heap number                   |
| 3 bits: record type (node pointer, infimum, supremum, etc.) |
| 2 bytes: relative pointer to next record in page |
ORIGIN (start of fields)
| first field data |
...
| last field data  |
```

1. **Relative vs. absolute “next record pointer”**: For new-style, it’s typically a “relative pointer” to the next record (the distance in bytes from one record to the next), whereas old style had a 2-byte “absolute offset in the page.”  
2. **3 bits for record type**:  
   - `000` = normal user record  
   - `001` = node pointer (internal non-leaf B-tree record)  
   - `010` = “infimum” (the smallest possible record used as a page sentinel)  
   - `011` = “supremum” (the largest possible record used as a page sentinel)  
   - `1xx` = reserved  
3. **Variable-length column lengths**:  
   - If the column’s declared max length is <= 255, we store a 1-byte length.  
   - Otherwise, or for BLOB columns, we might need 2 bytes. The top bits can indicate “externally stored.”  
4. **Null flags**:  
   - Immediately below the length bytes is a bitmap with 1 bit per nullable column. If that bit is 1, then the column is physically absent from the data portion (SQL `NULL`).

### Example: Let’s say we have a record with columns `("AB", (SQL NULL), "CDEF")`

If it’s ROW_FORMAT=COMPACT:

1. For each variable-length column, we store a length byte (or two bytes for bigger columns).  
2. If the second column is NULL, we do **not** store any length byte for that column; we just set the bit in the “null flags” bitmap.  
3. Suppose columns 1 & 3 are each < 128 in length, so we use a single byte for each length:  
   - col1 length = 2 → stored as `0x02`,  
   - col2 is NULL → no length byte, set the null-flag bit,  
   - col3 length = 4 → stored as `0x04`.  

   The “length area” near the top of the record might look like `[ 0x04, 0x02 ]`. Notice they store them in **reverse** order of columns. Then comes the null-flag byte `[ 0x02? ]`, etc.  
4. The actual data that follows is just: `"AB"` then `"CDEF"`.

---

## 3. Canonical Coordinates

In the code comments, there is a notion of a “canonical string” representation of a record. This is a conceptual device for comparing records or dealing with prefix lengths. 

### How it works conceptually

1. **Concatenate the bytes of each field in order**. SQL-NULL fields become an **empty** string.  
2. **After each field** in this conceptual string, insert a special marker:  
   - `<FIELD-END>` if the field had data,  
   - `<NULL-FIELD-END>` if the field was SQL-NULL.  
3. This extended string is the “canonical representation.” You can measure or compare positions in this string in a consistent manner, ignoring the intricacies of how data is actually stored on disk.

### Example 1: `( "AA", SQL-NULL, "BB", "" )`
- Field 1 = `"AA"` → contributes `"AA"` + `<FIELD-END>`  
- Field 2 = `SQL-NULL` → contributes `""` + `<NULL-FIELD-END>`  
- Field 3 = `"BB"` → contributes `"BB"` + `<FIELD-END>`  
- Field 4 = `""` (empty string) → contributes `""` + `<FIELD-END>`  

Hence, the canonical string:

```
"AA" <FIELD-END> 
      <NULL-FIELD-END> 
"BB" <FIELD-END> 
      <FIELD-END>
```
If we label the special markers with single-letter placeholders, say:  
- `<FIELD-END>` = `F`  
- `<NULL-FIELD-END>` = `N`  

Then the canonical string is:  
```
"AA" F N "BB" F F
```
Length = 2 (for "AA") + 1 (F) + 1 (N) + 2 (for "BB") + 1 (F) + 1 (F) = 8 total “characters,” if you imagine each marker as 1.

### Example 2 (from the comment):  
`("AA", SQL-NULL, "BB", "C")` and `("AA", SQL-NULL, "B", "C")`

- First record’s canonical representation:  
  `AA <FIELD-END> <NULL-FIELD-END> BB <FIELD-END> C <FIELD-END>`
  
- Second record’s canonical representation:  
  `AA <FIELD-END> <NULL-FIELD-END> B <FIELD-END> C <FIELD-END>`

The **maximal common prefix** is:
```
AA<FIELD-END><NULL-FIELD-END>B
```
because for the first record, the next character is “B,” but for the second record, that next character is `<FIELD-END>` vs. “B.” So the matching stops right after the “B” in the second record’s third field. They mention the canonical length of that prefix is `5` if you count each real character + marker.

In simpler “ASCII placeholder” terms:
- First record: `AA F N BB F C F`
- Second record: `AA F N B F C F`

They both match: `AA F N B` (4 real characters in "AAB" + 3 markers so far, depending on how we count “B” plus the marker). The code’s precise mention of “5” is because they specifically measure each letter or marker as one position in the conceptual indexing.  

---

## 4. Why This Matters

1. **Comparisons / Prefixes**:  
   MySQL can do prefix comparison (e.g. partial key matches), so InnoDB needs a consistent way to see how far two records match. That is where the canonical coordinate concept helps—especially for variable-length or null fields.
2. **Searching in the B-tree**:  
   InnoDB’s B-tree code may compare records or part of them. Instead of handling the nuances of offsets, null bits, etc., the “canonical” approach is a conceptual way to unify comparisons.
3. **Debugging and Diagnostics**:  
   The comments in `rem/rem0rec.c` clarify how to parse the raw bytes you see on disk or in memory for an InnoDB index page. Tools like the MySQL “page inspector” (and various InnoDB parsing utilities) rely on these definitions to display the correct field values.

---

## 5. Summary

- **Old style**: one- or two-byte offsets, each offset holding a bit for SQL-NULL, plus possibly an “extern” bit. Extra bits in the record header specify how many fields, how large each offset is, etc.  
- **New style (compact)**: length bytes for variable-length fields, a bitmap for null fields, a 3-bit record type, and a 2-byte relative pointer for the next record. This generally saves space, especially for variable-length columns.  
- **Canonical Coordinates**: a conceptual tool for treating all fields as a single string with markers after each field (or null field) so that prefix operations and comparisons become easier to define.  

Hence, the examples:

- `("AA", SQL-NULL, "BB", "")` → `"AA<FIELD-END><NULL-FIELD-END>BB<FIELD-END><FIELD-END>"`
- The partial matching example in the comments → shows how we find the length of the common prefix in that canonical string sense.  

All of this is crucial to how **InnoDB** manages, indexes, and compares records internally.

# rem0rec.c

Below is a **detailed, function-by-function walkthrough** of the relevant code in **`rem/rem0rec.c`**, **focusing** on how each function works internally, and **relating** it to the **physical-record concepts** (old-style vs. new-style) we already discussed. You can refer back to the previously explained record formats if you want a refresher on the structural details.

---

## 1. `rec_get_n_extern_new()`
```c
UNIV_INTERN
ulint
rec_get_n_extern_new(
/*=================*/
    const rec_t*   rec,   /*!< in: compact physical record */
    dict_index_t*  index, /*!< in: record descriptor */
    ulint          n)     /*!< in: number of columns to scan */
{
    const byte*    nulls;
    const byte*    lens;
    dict_field_t*  field;
    ulint          null_mask;
    ulint          n_extern;
    ulint          i;

    ut_ad(dict_table_is_comp(index->table));
    ut_ad(rec_get_status(rec) == REC_STATUS_ORDINARY);
    ut_ad(n == ULINT_UNDEFINED || n <= dict_index_get_n_fields(index));

    if (n == ULINT_UNDEFINED) {
        n = dict_index_get_n_fields(index);
    }

    nulls = rec - (REC_N_NEW_EXTRA_BYTES + 1);
    lens = nulls - UT_BITS_IN_BYTES(index->n_nullable);
    null_mask = 1;
    n_extern = 0;
    i = 0;

    /* read the lengths of fields 0..n */
    do {
        ulint len;

        field = dict_index_get_nth_field(index, i);
        if (!(dict_field_get_col(field)->prtype & DATA_NOT_NULL)) {
            /* nullable field => read the null flag */

            if (UNIV_UNLIKELY(!(byte) null_mask)) {
                nulls--;
                null_mask = 1;
            }

            if (*nulls & null_mask) {
                null_mask <<= 1;
                /* No length is stored for NULL fields. */
                continue;
            }
            null_mask <<= 1;
        }

        if (UNIV_UNLIKELY(!field->fixed_len)) {
            /* Variable-length field: read the length */
            const dict_col_t* col = dict_field_get_col(field);
            len = *lens--;
            if (UNIV_UNLIKELY(col->len > 255)
                || UNIV_UNLIKELY(col->mtype == DATA_BLOB)) {
                if (len & 0x80) {
                    /* 1exxxxxxx xxxxxxxx */
                    if (len & 0x40) {
                        n_extern++;
                    }
                    lens--;
                }
            }
        }
    } while (++i < n);

    return(n_extern);
}
```
**Purpose**: For a **ROW_FORMAT=COMPACT** record, this function iterates through the **first** `n` columns and **counts** how many of them are stored externally (i.e., `BLOB` columns that exceed a certain size and are placed in external pages). It **does not** parse all offsets completely; rather, it only checks each field’s length bytes for the `extern` bit.

### How it works internally
1. **Check assumptions**: 
   - `dict_table_is_comp(index->table)` → the record is in “compact” format (new style).  
   - `rec_get_status(rec) == REC_STATUS_ORDINARY` → make sure it’s a normal user record (not infimum, supremum, or node pointer).  
   - `n == ULINT_UNDEFINED || n <= dict_index_get_n_fields(index)` → you can either pass a specific `n` columns or pass `ULINT_UNDEFINED` to consider all columns.
2. **Set up pointers**:
   - `nulls` → points to the **null-bit** array in the record header (the bits that say which fields are `NULL`).  
   - `lens` → points to the **length bytes** for variable-length columns, which appear *before* the `nulls` array (in reverse).
3. **Loop over each field**:
   - If the field can be `NULL` (`DATA_NOT_NULL` is not set), we check if it **is** `NULL` by inspecting the `nulls` bit. If it’s `NULL`, no length bytes are stored, so skip it.
   - If it’s not `NULL`, and the field is not of fixed length, we read the length byte(s) from `*lens--`.
   - If the length byte has the top bit (`0x80`) set, we might have a 2-byte length (the second byte is at `lens--` again). 
     - Then we check the “extern” bit (`0x40`) to see if the field is externally stored.
   - We increment `n_extern` if the extern bit is set.
4. **Return**: The count of externally stored columns among the first `n` fields.

---

## 2. `rec_init_offsets_comp_ordinary()`
```c
UNIV_INTERN
void
rec_init_offsets_comp_ordinary(
/*===========================*/
    const rec_t*       rec,    /*!< in: physical record in ROW_FORMAT=COMPACT */
    ulint              extra,  /*!< in: number of bytes to reserve between
                                the record header and the data payload
                                (usually REC_N_NEW_EXTRA_BYTES) */
    const dict_index_t* index, /*!< in: record descriptor */
    ulint*             offsets)/*!< in/out: array of offsets;
                                in: n=rec_offs_n_fields(offsets) */
{
    ulint    i      = 0;
    ulint    offs   = 0;
    ulint    any_ext= 0;
    const byte* nulls= rec - (extra + 1);
    const byte* lens = nulls - UT_BITS_IN_BYTES(index->n_nullable);
    dict_field_t* field;
    ulint    null_mask = 1;

#ifdef UNIV_DEBUG
    offsets[2] = (ulint) rec;
    offsets[3] = (ulint) index;
#endif /* UNIV_DEBUG */

    /* read the lengths of fields 0..n */
    do {
        ulint len;

        field = dict_index_get_nth_field(index, i);
        if (!(dict_field_get_col(field)->prtype & DATA_NOT_NULL)) {
            /* nullable field => read the null flag */

            if (UNIV_UNLIKELY(!(byte) null_mask)) {
                nulls--;
                null_mask = 1;
            }

            if (*nulls & null_mask) {
                null_mask <<= 1;
                /* No length is stored for NULL fields.
                We do not advance offs, and we set
                the length to zero and enable the
                SQL NULL flag in offsets[]. */
                len = offs | REC_OFFS_SQL_NULL;
                goto resolved;
            }
            null_mask <<= 1;
        }

        if (UNIV_UNLIKELY(!field->fixed_len)) {
            /* Variable-length field: read the length */
            const dict_col_t* col = dict_field_get_col(field);
            len = *lens--;
            if (UNIV_UNLIKELY(col->len > 255)
                || UNIV_UNLIKELY(col->mtype == DATA_BLOB)) {
                if (len & 0x80) {
                    /* 1exxxxxxx xxxxxxxx */
                    len <<= 8;
                    len |= *lens--;

                    offs += len & 0x3fff; 
                    if (UNIV_UNLIKELY(len & 0x4000)) {
                        ut_ad(dict_index_is_clust(index));
                        any_ext = REC_OFFS_EXTERNAL;
                        len = offs | REC_OFFS_EXTERNAL;
                    } else {
                        len = offs;
                    }

                    goto resolved;
                }
            }

            len = offs += len;
        } else {
            len = offs += field->fixed_len;
        }
resolved:
        rec_offs_base(offsets)[i + 1] = len;
    } while (++i < rec_offs_n_fields(offsets));

    *rec_offs_base(offsets) = (rec - (lens + 1)) | REC_OFFS_COMPACT | any_ext;
}
```
**Purpose**: This is a **specialized** function for creating the offsets array for a **compact (new-style)**, **ordinary** (leaf-level) record. It interprets the length bytes (and null bits) from the record header to compute the start/end offsets of each field.

### Key Points
- `nulls = rec - (extra + 1)` → This locates the bitmap of null flags in the record header area.  
- `lens = nulls - UT_BITS_IN_BYTES(index->n_nullable)` → This locates the “length bytes” region.  
- We iterate over fields [0..n-1], checking if each is `NULL` or not.  
- If it’s `NULL`, we mark the offset with `REC_OFFS_SQL_NULL`. If it’s an externally stored column, we mark it with `REC_OFFS_EXTERNAL`.  
- The offset we store in `rec_offs_base(offsets)[i+1]` is the **cumulative** offset from the record origin to the end of the field.  
- `any_ext` is used to note whether **any** field in the record was external (`REC_OFFS_EXTERNAL` bit). If so, we OR that bit into `rec_offs_base(offsets)[0]`.

---

## 3. `rec_init_offsets()`
```c
UNIV_STATIC
void
rec_init_offsets(
/*=============*/
    const rec_t*       rec,
    const dict_index_t* index,
    ulint*             offsets)
{
    ulint i = 0;
    ulint offs;

    rec_offs_make_valid(rec, index, offsets);

    if (dict_table_is_comp(index->table)) {
        /* new-style (compact) record path */
        ...
        switch (UNIV_EXPECT(status, REC_STATUS_ORDINARY)) {
        case REC_STATUS_INFIMUM:
        case REC_STATUS_SUPREMUM:
            ...
            break;
        case REC_STATUS_NODE_PTR:
            ...
            break;
        case REC_STATUS_ORDINARY:
            rec_init_offsets_comp_ordinary(...);
            return;
        }
        ...
    } else {
        /* old-style record path */
        ...
        if (rec_get_1byte_offs_flag(rec)) {
            /* read offsets from 1-byte array */
        } else {
            /* read offsets from 2-byte array */
        }
    }
}
```
**Purpose**: This is a **unified** function that determines how to parse offsets for both old-style and new-style records. It checks the row format, whether the record is ordinary, node pointer, infimum, or supremum, and then dispatches:

- For **compact** (`dict_table_is_comp(...) == TRUE`), calls one of:
  - `rec_init_offsets_comp_ordinary()`
  - or handles special “node pointer,” “infimum/supremum” logic
- For **old-style**, it decodes either 1-byte or 2-byte offsets from the record header, sets the `REC_OFFS_SQL_NULL` or `REC_OFFS_EXTERNAL` bits accordingly.

In all cases, it ends with a fully populated `offsets[]` array that says, for each column, “where does the field end in the record?”

---

## 4. `rec_get_offsets_func()`
```c
UNIV_INTERN
ulint*
rec_get_offsets_func(
/*=================*/
    const rec_t*        rec,
    const dict_index_t* index,
    ulint*              offsets,
    ulint               n_fields,
    mem_heap_t**        heap,
    const char*         file,
    ulint               line)
{
    ulint n;
    ulint size;

    ...
    /* Decide how many fields are actually in this record: */
    if (dict_table_is_comp(index->table)) {
        switch (UNIV_EXPECT(rec_get_status(rec), REC_STATUS_ORDINARY)) {
        case REC_STATUS_ORDINARY:
            n = dict_index_get_n_fields(index);
            break;
        case REC_STATUS_NODE_PTR:
            n = dict_index_get_n_unique_in_tree(index) + 1;
            break;
        case REC_STATUS_INFIMUM:
        case REC_STATUS_SUPREMUM:
            n = 1;
            break;
        default:
            ut_error;
            return(NULL);
        }
    } else {
        n = rec_get_n_fields_old(rec);
    }

    if (UNIV_UNLIKELY(n_fields < n)) {
        n = n_fields;
    }

    size = n + (1 + REC_OFFS_HEADER_SIZE);

    /* Possibly allocate or reuse an offsets array from a heap */
    if (UNIV_UNLIKELY(!offsets)
        || UNIV_UNLIKELY(rec_offs_get_n_alloc(offsets) < size)) {
        if (UNIV_UNLIKELY(!*heap)) {
            *heap = mem_heap_create_func(size * sizeof(ulint),
                                         MEM_HEAP_DYNAMIC,
                                         file, line);
        }
        offsets = mem_heap_alloc(*heap, size * sizeof(ulint));
        rec_offs_set_n_alloc(offsets, size);
    }

    rec_offs_set_n_fields(offsets, n);
    rec_init_offsets(rec, index, offsets);
    return(offsets);
}
```
**Purpose**:
- A **convenience** function for higher-level code to get the offsets array for a record. 
- It figures out how many fields to read (`n`), checks if you passed a smaller `n_fields`, then calls `rec_init_offsets()` to do the actual parse.
- It also manages memory for the offsets array, possibly reusing one you provided, or allocating from a heap if needed.

### Key Points
- `n = dict_index_get_n_fields(index)` for an ordinary record in a compact table, or `rec_get_n_fields_old(rec)` for an old-style table.  
- If you pass `n_fields` that is smaller, it means “only parse offsets for the first `n_fields` columns.”  
- The final call to `rec_init_offsets()` actually populates the data.

---

## 5. `rec_get_offsets_reverse()`
```c
UNIV_INTERN
void
rec_get_offsets_reverse(
/*====================*/
    const byte*       extra,
    const dict_index_t* index,
    ulint             node_ptr,
    ulint*            offsets)
{
    ...
    /*
      This function is basically the mirror logic, but reading length
      bytes in reverse order from 'extra' for a compact record.
      It's used in certain specialized situations (like prefix compression).
    */
}
```
**Purpose**: This is a **special** helper for reading offsets in reverse, often used by the InnoDB compression or special B-Tree logic. Instead of starting from the “top” of the record header, it starts from the “extra” bytes in **reverse**. Very similar logic to `rec_init_offsets_comp_ordinary()`, but reversed.

---

## 6. `rec_get_nth_field_offs_old()`
```c
UNIV_INTERN
ulint
rec_get_nth_field_offs_old(
/*=======================*/
    const rec_t* rec,
    ulint        n,
    ulint*       len)
{
    ulint os;
    ulint next_os;

    if (rec_get_1byte_offs_flag(rec)) {
        os = rec_1_get_field_start_offs(rec, n);
        next_os = rec_1_get_field_end_info(rec, n);
        if (next_os & REC_1BYTE_SQL_NULL_MASK) {
            *len = UNIV_SQL_NULL;
            return(os);
        }
        next_os = next_os & ~REC_1BYTE_SQL_NULL_MASK;
    } else {
        os = rec_2_get_field_start_offs(rec, n);
        next_os = rec_2_get_field_end_info(rec, n);
        if (next_os & REC_2BYTE_SQL_NULL_MASK) {
            *len = UNIV_SQL_NULL;
            return(os);
        }
        next_os = next_os & ~(REC_2BYTE_SQL_NULL_MASK | REC_2BYTE_EXTERN_MASK);
    }

    *len = next_os - os;
    return(os);
}
```
**Purpose**: For **old-style** records (1-byte or 2-byte offsets), this directly computes “the offset from the record origin to field `n`” and sets `*len` to the length of that field—**or** `UNIV_SQL_NULL` if it’s marked as `NULL`.

**Key Steps**:
- If 1-byte offsets:
  - Use `rec_1_get_field_start_offs(...)` to find where that field starts.
  - Use `rec_1_get_field_end_info(...)` to see where it ends, also checking the null bit.  
- If 2-byte offsets:
  - Similar, but calling `rec_2_get_field_start_offs(...)` etc., which has a second bit for “extern” storage.  
- Subtract start from end to get the length.

---

## 7. `rec_get_converted_size_comp_prefix()` and `rec_get_converted_size_comp()`
These two functions **calculate** how large a **compact-format** record would be if we built it from a data tuple (`dfield_t` array).

- `rec_get_converted_size_comp_prefix(...)` calculates the size of the “payload” (the actual data fields) plus the “extra size” needed in the record header (length bytes, null flags, etc.). 
- `rec_get_converted_size_comp(...)` calculates the total size for a record, factoring in whether it’s a node pointer record, or infimum/supremum, etc.

**Purpose**: This is used when building a brand-new record from user data: we need to figure out how many bytes of record header we will need, then how many bytes for data, etc.

---

## 8. `rec_set_nth_field_null_bit()` and `rec_set_nth_field_sql_null()`
```c
UNIV_INTERN
void
rec_set_nth_field_null_bit(rec_t* rec, ulint i, ibool val)
{
    // Sets just the bit. old-style: handle 1-byte vs 2-byte offset arrays.
}

UNIV_INTERN
void
rec_set_nth_field_sql_null(rec_t* rec, ulint n)
{
    // Actually writes the special "SQL NULL" pattern and sets the bit
    // in the offset array.
}
```
**Purpose**: For **old-style** records only. These manipulate the “null bit” in the offset array. `rec_set_nth_field_sql_null()` also writes a SQL-null marker (which is usually a sequence of zero bytes) into the data portion to mark it physically as `NULL`.

---

## 9. `rec_convert_dtuple_to_rec_old()` vs. `rec_convert_dtuple_to_rec_comp()`
These are **core** routines to create a raw physical record from a `dtuple_t` (an internal InnoDB structure representing a row or partial row).

### `rec_convert_dtuple_to_rec_old()`
1. Takes a buffer `buf` where we will write the record, plus the `dtuple_t`.
2. Figures out if we can use 1-byte offsets or need 2-byte offsets, whether fields are null or external, etc.
3. Writes the data fields into the record one by one.  
4. Sets the offset array accordingly (with or without `REC_1BYTE_SQL_NULL_MASK`, `REC_2BYTE_SQL_NULL_MASK`, `REC_2BYTE_EXTERN_MASK`).
   
### `rec_convert_dtuple_to_rec_comp()`
Similar, but **for** ROW_FORMAT=COMPACT. We do:
- Write the variable-length column length bytes (checking whether we need one or two bytes).
- Write the null-bit map for any nullable columns.
- Copy the actual field data into the record payload.

---

## 10. `rec_convert_dtuple_to_rec()` (The main entry point)
```c
UNIV_INTERN
rec_t*
rec_convert_dtuple_to_rec(
    byte*            buf,
    const dict_index_t* index,
    const dtuple_t*  dtuple,
    ulint            n_ext)
{
    rec_t* rec;

    if (dict_table_is_comp(index->table)) {
        rec = rec_convert_dtuple_to_rec_new(buf, index, dtuple);
    } else {
        rec = rec_convert_dtuple_to_rec_old(buf, dtuple, n_ext);
    }

#ifdef UNIV_DEBUG
    // Validate the record we just created.
#endif
    return(rec);
}
```
**Purpose**: Decides based on table format (compact or old-style) which conversion function to call. This is the usual function that other code calls to turn a `dtuple_t` into an actual record in memory.

---

## 11. `rec_copy_prefix_to_dtuple()` and `rec_copy_prefix_to_buf()`
These let us **copy only the first N fields** of a record, either:
- **To a data tuple** (`dtuple_t`), or
- **To another physical record** in a buffer.

This is useful in partial key operations, prefix comparisons, or constructing partial keys for search.

---

## 12. `rec_validate()` / `rec_validate_old()`
Finally, the **validation** routines:

- `rec_validate_old()`: Checks an old-style record’s offsets, ensuring the sum of field lengths matches the record length, etc.
- `rec_validate()`: 
  1. Checks how many fields are in the record.  
  2. Iterates through each field (using the offsets array) to ensure no invalid lengths.  
  3. If it’s old-style, calls `rec_validate_old()`.  

They help detect any corruption or mismatch in the record’s internal structure.

---

## Putting It All Together

1. **Reading**:  
   - Functions like `rec_init_offsets()`, `rec_get_offsets_func()`, and `rec_get_nth_field_offs_old()` are used to parse an **existing** physical record in memory and figure out “where does each column begin and end?” or “which columns are null?” or “is there a BLOB column stored externally?”  

2. **Writing**:  
   - Functions like `rec_convert_dtuple_to_rec_old()` or `rec_convert_dtuple_to_rec_comp()` take a **logical** row (`dtuple_t`) and write out all the correct header bytes, length bytes, null bits, offsets, etc. into a **physical record** layout.

3. **Validation**:  
   - `rec_validate()` ensures that the record in memory is not corrupt, that the field offsets make sense, and that the total data length matches what the offset array implies.

4. **Partial Access**:  
   - Functions like `rec_copy_prefix_to_buf()` or `rec_copy_prefix_to_dtuple()` allow partial extraction of fields (for prefix comparisons or partial key building).

In short, **this file** (`rem0rec.c`) is a **crucial** part of the **row/record manager**. It hides the complexity of **old style** vs. **compact** row format, and it provides the building blocks so that higher-level InnoDB code (in `row0ins.c`, `row0upd.c`, `btr0cur.c`, etc.) can easily read or write the correct bytes into pages.

# rem0cmp.c

Below is a **detailed, function-by-function walkthrough** of the **`rem0cmp.c`** file in InnoDB, focusing on **how each function works internally** and **relating** it to the **record comparison concepts** in InnoDB. As you read, you can refer back to the notion of *alphabetical ordering*, *field lengths/padding*, *SQL NULL ordering*, and *collation* details described in the file’s top comments.

---

## Overall Purpose of `rem0cmp.c`

This file is all about **comparing**:

1. **Data fields** and **physical records** in InnoDB**:
   - The code implements an “alphabetical order” (or lexicographical order) for rows on disk.
   - It handles corner cases like:
     - **SQL NULL** comparison (NULL is defined to be “smaller” or “bigger,” depending on version, but in MySQL/InnoDB it’s typically considered *smaller* or *largest*, depending on code paths, but here the comments say “Finally, the SQL null is bigger than any other value.”—there are small nuances).
     - **Padding** for CHAR vs. VARCHAR fields (when needed).
     - **Signed vs. unsigned integer** comparisons.
     - **Collation transformations** (for example, `latin1` or other charsets).
2. **Data tuples** (`dtuple_t`) to **physical records** (`rec_t`), partial comparisons, prefix checks, etc.

You’ll find routines to:
- Compare a **tuple** and a **record**: `cmp_dtuple_rec()`, `cmp_dtuple_is_prefix_of_rec()`.
- Compare **two physical records** directly: `cmp_rec_rec_simple()`, `cmp_rec_rec_with_match()`.
- Compare **individual data fields** with or without collation: `cmp_data_data_slow()`, `cmp_whole_field()`.
- Check if two columns are “comparable”: `cmp_cols_are_equal()`.
- Perform debug checks: `cmp_debug_dtuple_rec_with_match()`.

---

## 1. High-Level “Alphabetical Order” Concept

In the comment block near the top, it explains:

> The records are put into alphabetical order in the following way: let F be the first field where two records disagree...

- In practice, when comparing two records, we scan each field in order. If all fields match up to field `F`, we compare the two field-`F` values.  
- If the values differ in some byte, we pick the order based on that differing byte (after optional “collation transformations”).  
- If the fields are the same in all corresponding bytes but differ in length (and the type is **paddable**, e.g. CHAR), we treat the shorter field as if it’s padded (with spaces or 0x00 depending on collation rules).  
- If the type is *not* paddable (e.g. a variable-length type like `VARCHAR`?), then the longer field is considered bigger.  
- **NULL** is considered “bigger” than all non-null values (the code sometimes inverts this, so we have to see which function uses which logic).

That’s the overarching principle behind all the compare functions.

---

## 2. `cmp_collate()`

```c
UNIV_INLINE
ulint
cmp_collate(
/*========*/
    ulint code)  /*!< in: code of a character stored in database record */
{
    //return((ulint) srv_latin1_ordering[code]);
    /* FIXME: Default to ASCII */
    return(code);
}
```

- **Purpose**: This function takes a character code (`code`) and transforms it according to a collation (like `latin1`). In the actual MySQL/InnoDB code, `srv_latin1_ordering[]` might reorder ASCII codes so that, e.g., uppercase vs. lowercase have some order.  
- Here, it just returns `code` itself (the “FIXME” shows that it’s a stub default).  
- In real usage, if your column’s collation is something like `latin1_swedish_ci`, InnoDB might reorder or fold the characters. But for simplicity, this function is effectively a “no-op” in this source snapshot.

---

## 3. `cmp_cols_are_equal()`
```c
UNIV_INTERN
ibool
cmp_cols_are_equal(
    const dict_col_t* col1,
    const dict_col_t* col2,
    ibool             check_charsets)
{
    ...
}
```
- **Purpose**: Checks if two columns (`col1` and `col2`) are considered “equivalent enough” to be comparable.  
- **Key checks**:
  1. If both are **non-binary string** types, we can compare them only if they share the same charset/collation (unless `check_charsets` is FALSE).  
  2. If both are **binary** string types, we allow comparison.  
  3. If both are integer, check if they’re both signed or both unsigned. Signed vs. unsigned differ in the raw on-disk format.  
  4. For other types, we check if `col1->mtype == col2->mtype` and, for fixed integer, if the lengths match.

**Return**: `TRUE` if they are “comparable,” `FALSE` otherwise.

---

## 4. `cmp_whole_field()`
```c
UNIV_STATIC
int
cmp_whole_field(
    void*       cmp_ctx,
    ulint       mtype,
    ib_u16_t    prtype,
    const byte* a,
    unsigned    a_length,
    const byte* b,
    unsigned    b_length)
{
    ...
}
```
- **Purpose**: Compare two data fields “as a whole,” used for types that can’t be easily compared byte-by-byte with padding. Examples:
  - `DATA_DECIMAL`, `DATA_DOUBLE`, `DATA_FLOAT`, `DATA_BLOB`, or a client type that requires calling `ib_client_compare(...)`.
- **Implementation**:
  - For `DATA_DECIMAL`, it does manual handling of any leading `'-'` sign, leading zeros, etc.  
  - For `DATA_DOUBLE` and `DATA_FLOAT`, we read them as machine floats/doubles (`mach_double_read()`, etc.) and compare numerically.  
  - For `DATA_BLOB`, or “client” types, we pass them to `ib_client_compare(...)`, which is an external function that does a custom comparison depending on the client type.  
- **Return**: `1` if `a > b`, `0` if equal, `-1` if `a < b`.

---

## 5. `cmp_data_data_slow()`
```c
UNIV_INTERN
int
cmp_data_data_slow(
    void*       cmp_ctx,
    ulint       mtype,
    ulint       prtype,
    const byte* data1,
    ulint       len1,
    const byte* data2,
    ulint       len2)
{
    ...
}
```
- **Purpose**: Compare two fields (`data1`, `data2`) with known type `mtype`, “the slow path.” If the field is a float/double, or a BLOB with a non-latin1 collation, it delegates to `cmp_whole_field(...)`. Otherwise, it tries to compare them byte-by-byte with optional `cmp_collate()` transformations and possible padding characters if one is shorter.
- **Key Steps**:
  1. **Check SQL NULL**: if either is `UNIV_SQL_NULL`, we decide who’s bigger.  
  2. If it’s a floating type or certain BLOB conditions, call `cmp_whole_field()`.  
  3. Otherwise, do a loop over each byte:
     - If we run out of bytes in one field but not the other, we might try to “pad” the shorter field (e.g. with space, depending on the type).  
     - Compare after possibly applying `cmp_collate(...)`.  
     - If we find a difference, return `1` or `-1`.
     - If we exhaust both fields, return `0` if they matched.

---

## 6. `cmp_dtuple_rec_with_match()`
```c
UNIV_INTERN
int
cmp_dtuple_rec_with_match(
    void*       cmp_ctx,
    const dtuple_t* dtuple,
    const rec_t*   rec,
    const ulint*   offsets,
    ulint*     matched_fields,
    ulint*     matched_bytes)
{
    ...
}
```
- **Purpose**: Compare a **data tuple** (`dtuple_t`) to a **physical record** (`rec_t`) up to some number of fields, or until a mismatch is found.  
- **Key Points**:
  1. `matched_fields` and `matched_bytes` let you continue partial comparisons (like “how many fields have we already matched so far?”). This is used in certain B-tree operations that do incremental comparisons.  
  2. We loop through fields, checking if the record’s field is externally stored. If so, we bail out with `ret = 0` (some code simply can’t handle partial compare of an external field).  
  3. For each field, if both are `SQL NULL`, they match. If one is `NULL` and the other not, we resolve order.  
  4. If the type is complicated (float, decimal, etc.), we call `cmp_whole_field()`.  
  5. Otherwise, we do the typical byte-by-byte comparison, applying `cmp_collate()`. We stop as soon as a difference is found or we run out of data in one field.  
  6. Update `matched_fields` and `matched_bytes` as we go.  
- **Return**: 1 if `dtuple` is bigger, 0 if they match up to an externally stored field or are equal up to the common prefix, -1 if `dtuple` is smaller.

### `cmp_dtuple_rec()`
```c
UNIV_INTERN
int
cmp_dtuple_rec(
    void*           cmp_ctx,
    const dtuple_t* dtuple,
    const rec_t*    rec,
    const ulint*    offsets)
{
    ulint matched_fields = 0;
    ulint matched_bytes  = 0;

    return cmp_dtuple_rec_with_match(
        cmp_ctx, dtuple, rec, offsets,
        &matched_fields, &matched_bytes);
}
```
- **Purpose**: A simpler wrapper that calls `cmp_dtuple_rec_with_match()` with `matched_fields=0` and `matched_bytes=0`, i.e., from scratch.  
- **Returns** `1`, `0`, or `-1`.

---

## 7. `cmp_dtuple_is_prefix_of_rec()`
```c
UNIV_INTERN
ibool
cmp_dtuple_is_prefix_of_rec(
    void*           cmp_ctx,
    const dtuple_t* dtuple,
    const rec_t*    rec,
    const ulint*    offsets)
{
    ...
}
```
- **Purpose**: Checks if the entire **`dtuple`** is a prefix of the **`rec`**. The last field in the `dtuple` is allowed to be a prefix of the corresponding field in `rec`.  
- **How**:
  1. Compare `dtuple` and `rec` using `cmp_dtuple_rec_with_match()`.  
  2. If we matched all fields in `dtuple` fully, that’s obviously a prefix.  
  3. Or if we matched up through the second-to-last field, and for the last field, we matched all its bytes (which might be shorter than the record’s field), that also qualifies as a prefix.  
- **Return**: `TRUE` if prefix, `FALSE` otherwise.

---

## 8. `cmp_rec_rec_simple()`
```c
UNIV_INTERN
int
cmp_rec_rec_simple(
    const rec_t*       rec1,
    const rec_t*       rec2,
    const ulint*       offsets1,
    const ulint*       offsets2,
    const dict_index_t* index)
{
    ...
}
```
- **Purpose**: Compare **two physical records** (`rec1` and `rec2`) that have the **same number of columns** and none are externally stored. It’s a simpler scenario.  
- **Key Steps**:
  1. We retrieve each field from both records, check if either is `NULL`. If both non-null, we do “byte by byte with optional collation transform,” or if it’s a “float/double/other special,” we call `cmp_whole_field()`.  
  2. We do this only for the first `n_uniq` columns (the number of “unique” columns in the index). Once we find a mismatch, we return `1` or `-1`. If we finish all fields, they’re considered equal (`0`).

---

## 9. `cmp_rec_rec_with_match()`
```c
UNIV_INTERN
int
cmp_rec_rec_with_match(
    const rec_t*   rec1,
    const rec_t*   rec2,
    const ulint*   offsets1,
    const ulint*   offsets2,
    dict_index_t*  index,
    ulint*         matched_fields,
    ulint*         matched_bytes)
{
    ...
}
```
- **Purpose**: A more general version of comparing two **physical records**. Also supports partial matching up to `matched_fields/bytes`.  
- Very similar logic to `cmp_dtuple_rec_with_match()`, but for record vs. record:
  1. If a field is externally stored in either record, we bail out with a partial comparison result (often `0`).  
  2. If fields differ in `NULL`ness, we choose an order.  
  3. If they are floats/doubles/complex, do `cmp_whole_field()`. Otherwise, compare byte by byte with optional collation.  
  4. Update `matched_fields` and `matched_bytes`.  
- **Return**: `1` if `rec1` > `rec2`, `-1` if `rec1` < `rec2`, `0` if they’re equal or we only matched a partial prefix.

---

## 10. `cmp_debug_dtuple_rec_with_match()` (Only in `#ifdef UNIV_DEBUG`)
```c
#ifdef UNIV_DEBUG
UNIV_STATIC
int
cmp_debug_dtuple_rec_with_match(
    void*           cmp_ctx,
    const dtuple_t* dtuple,
    const rec_t*    rec,
    const ulint*    offsets,
    ulint*          matched_fields)
{
    ...
}
#endif
```
- **Purpose** (in debug builds): It’s essentially a “checking” version of `cmp_dtuple_rec_with_match()`. It verifies the partial comparison logic is consistent. If it finds an externally stored field, it returns `0`.  
- Used only in `UNIV_DEBUG` builds as an additional assertion or cross-check.

---

## Putting It All Together

1. **Column-level** comparisons:
   - `cmp_collate()` for applying a collation to a single character code.
   - `cmp_whole_field()` and `cmp_data_data_slow()` for general field-level comparisons (including numeric conversions for float/double/decimal).
   - `cmp_cols_are_equal()` to check if two columns are “comparable.”

2. **Record-level** comparisons:
   - `cmp_dtuple_rec()` or `cmp_dtuple_rec_with_match()` to compare a *logical data tuple* to a *physical record*.
   - `cmp_rec_rec_simple()` or `cmp_rec_rec_with_match()` to compare *two physical records* directly.
   - `cmp_dtuple_is_prefix_of_rec()` specifically for checking if a data tuple is a prefix of a record.

3. **Debug code** (`cmp_debug_dtuple_rec_with_match()`) ensures that partial-match logic is correct, especially when dealing with external fields or NULL values.

4. **B-tree usage**: The B-tree code in InnoDB calls these comparison functions to determine if a new record should go to the left or right of an existing record, or if two records match on a unique key, etc.

Hence, **`rem0cmp.c`** forms the backbone of InnoDB’s **record-comparison** logic, bridging the raw byte data on disk (records) with the higher-level typed data (tuples).

# rem0rec.ic

Below is a **detailed explanation** of the inline “helper” functions in **`rem0rec.ic`** and how they fit into the **InnoDB page** structure. These functions are **low-level utilities** for interpreting and modifying the bits of a record header, determining the next record pointer on a page, reading or setting the number of fields, the “heap number,” info bits, etc. Ultimately, they help InnoDB manage the **physical layout of records** within a **page**.

---

## How It Relates to the Page Concept

In InnoDB, **records** are physically stored on **pages** (often 16 KB). Each record has a **header area** that includes:

- **Pointers** to the next/previous records.
- **Flags** (e.g., for record type, if it’s deleted, how many fields, how many records it “owns,” etc.).
- **Offset arrays** for columns (especially in old style).

All of these helper functions:

1. **Read/Write 1-byte or 2-byte bit fields** at the correct offset from the **record origin**.  
2. **Compute the “next record” pointer** or the offset in the page, enabling you to traverse the page’s linked list of records.  
3. **Deal with old-style vs. compact (new-style) record formats**, which place bits differently in the record header.

Because pages store many records, these inline functions ensure we handle the **on-page** offsets and flags precisely. They help manipulate records **in-place** on a page without rewriting the entire record.

---

## 1. Bit-Field Readers/Writers

### `rec_get_bit_field_1()` / `rec_set_bit_field_1()`
- **Purpose**: Get or set a bit field that resides **within a single byte** in the record header.
- **Signature**:
  ```c
  UNIV_INLINE ulint rec_get_bit_field_1(const rec_t* rec, ulint offs, ulint mask, ulint shift);
  UNIV_INLINE void  rec_set_bit_field_1(rec_t* rec, ulint val, ulint offs, ulint mask, ulint shift);
  ```
- **Context**: In old-style records, there are specific 1-byte fields for “short flag,” “number of records owned,” “info bits,” etc. In new-style records, we also have single-byte fields for “heap no.,” “n_owned,” etc.  
- **How**: 
  - `offs` says *how far* (in bytes) from the record origin we move “upwards” (since the record origin is usually the start of the data area; these fields are stored *before* that).  
  - `mask` and `shift` specify which bits in that byte we want.  
  - For reading, we do `mach_read_from_1(...) & mask >> shift`.  
  - For writing, we read the original byte, clear out the bits in `mask`, and then OR in `(val << shift)`.

### `rec_get_bit_field_2()` / `rec_set_bit_field_2()`
- **Purpose**: Same as above, but for **2-byte** fields in the header (e.g., old-style `n_fields`, `heap_no`, or “next record pointer” in old style).
- **Signature**:
  ```c
  UNIV_INLINE ulint rec_get_bit_field_2(const rec_t* rec, ulint offs, ulint mask, ulint shift);
  UNIV_INLINE void  rec_set_bit_field_2(rec_t* rec, ulint val, ulint offs, ulint mask, ulint shift);
  ```
- **Example**: In old style, the record uses 2 bytes to store “n_fields” plus bits for the “short flag.” We can mask off the relevant bits.

**Relation to page**: The record data is **within** a single page. These offsets are **within** that page. For old style, the record pointer is typically an absolute offset within the page. For new style (compact), it’s a “relative offset” that can “wrap” in a 16-bit field.

---

## 2. Next Record Pointers

### `rec_get_next_ptr_const()` / `rec_get_next_ptr()`
- **Purpose**: Returns a pointer (in memory) to the “next record” on the same page, or `NULL` if none.  
- **Logic**:
  - Old style: The next record pointer is a **2-byte absolute offset** from the page origin.  
  - New style (compact): The next record pointer is a **relative** 2-byte offset that can wrap around if the page size is 32 KB or 64 KB, etc.  
- **Implementation**:
  - Reads those 2 bytes. If it’s `0`, no next record. If not `0`, we add that offset to our current record’s address. (For new-style, we do `(current_offset + relative_offset) mod PAGE_SIZE`.)
  - `rec_get_next_ptr_const()` returns a `const rec_t*`, while `rec_get_next_ptr()` returns a non-const `rec_t*`.

### `rec_get_next_offs()`
- **Purpose**: Similar to above, but returns just the numeric **page offset** of the next record (or 0 if none).  
- **Example**: If you want to quickly see which offset in the page is next.  

### `rec_set_next_offs_old()` / `rec_set_next_offs_new()`
- **Purpose**: Write the “next record offset” in old/new style.  
- **Old style**: Directly store `mach_write_to_2(rec - REC_NEXT, next);`  
- **New style**: Must convert an absolute offset to the “relative offset” field. That often involves `(next - current_rec_offset) & 0xFFFF`.  

**Relation to page**: The entire page is effectively a ring buffer of 16 KB or up to 64 KB, so the new-style pointer can “wrap around.” This is how InnoDB’s **on-page** linked list of records is maintained.

---

## 3. Old-Style vs. New-Style Fields (Number of Fields, Owned, Heap Number, etc.)

### `rec_get_n_fields_old()` / `rec_set_n_fields_old()`
- **Purpose**: In **old-style** records, the 2-byte area in the header encodes the number of fields in the record.  
- **Implementation**: Uses `rec_get_bit_field_2()` / `rec_set_bit_field_2()` with the `REC_OLD_N_FIELDS_MASK` etc.

### `rec_get_n_fields()` (general)
```c
UNIV_INLINE
ulint
rec_get_n_fields(const rec_t* rec, const dict_index_t* index);
```
- **Purpose**: For a record in either old or new format, determine how many fields it has.  
- **Logic**:
  1. If **old style**, read `rec_get_n_fields_old(rec)`.  
  2. If **new style**, check the `rec_get_status(rec)`:
     - If `REC_STATUS_ORDINARY`, it’s `dict_index_get_n_fields(index)`.
     - If `REC_STATUS_NODE_PTR`, it’s `dict_index_get_n_unique_in_tree(index) + 1`.  
     - If `REC_STATUS_INFIMUM` or `REC_STATUS_SUPREMUM`, that record has 1 field.

**Relation to page**: Each **on-page record** has a “heap number” and “n_fields” to manage how the record is placed in the **page’s “row heap.”**

### `rec_get_n_owned_old()`, `rec_set_n_owned_old()`, `rec_get_n_owned_new()`, `rec_set_n_owned_new()`
- **Purpose**: The “owned” concept is used by InnoDB for “directory records” that can “own” subsequent records that are physically consecutive. This is part of InnoDB’s old page organization.  
- **Implementation**: For old style, it’s a 4-bit field in the record header; for new style, also 4 bits, but in a different location.  
- **Example**: Some records (like “directory slots”) own 4 or 8 subsequent records in the “page organization.” If you want to re-balance a page, you can increment or decrement how many are “owned” by a certain directory record.

### `rec_get_heap_no_old()`, `rec_set_heap_no_old()`, `rec_get_heap_no_new()`, `rec_set_heap_no_new()`
- **Purpose**: The “heap number” is the order in which the record was placed in the page’s “row heap.” It’s used for ordering or referencing the record.  
- **Implementation**: Old style stores it in 2 bytes with some bits. New style in a slightly different layout.

---

## 4. Info Bits, Status Bits, and Delete Mark

### `rec_get_info_bits()` / `rec_set_info_bits_old()` / `rec_set_info_bits_new()`
- **Purpose**: The “info bits” are 4 bits that store flags like “record is delete-marked,” “this is the min record,” etc.  
- **Implementation**:  
  - Old style: In the same byte as the “n_owned.”  
  - New style: In the byte after the “heap number.”  

### `rec_get_status()` and `rec_set_status()`
- **Purpose**: **New-style** records have “status bits” that define if the record is **ordinary**, **node pointer**, **infimum**, or **supremum**, or reserved.  
- **Implementation**: A 3-bit field in new-style records.  

### `rec_get_deleted_flag()` / `rec_set_deleted_flag_old()` / `rec_set_deleted_flag_new()`
- **Purpose**: If a record is “deleted,” InnoDB sets a bit in the **info bits**. This is the “delete-mark” bit. The row might still physically exist but is considered logically deleted.  
- **Implementation**: We read or set that particular bit in the 4-bit “info bits” region.

---

## 5. Offsets and Field Data (Old-Style)

### `rec_get_1byte_offs_flag()` / `rec_set_1byte_offs_flag()`
- **Purpose**: In old-style records, we can store each field’s offset as either 1 byte or 2 bytes if the record is small enough. This function checks (or sets) whether we’re using the 1-byte mode.  

### `rec_1_get_field_end_info()`, `rec_2_get_field_end_info()`
- **Purpose**: Return the “end offset” of a given field’s data from the record origin. The top bits can indicate `SQL NULL` or “extern stored.”  

### `rec_1_set_field_end_info()`, `rec_2_set_field_end_info()`
- **Purpose**: The inverse—write that “end offset” (with bits for `NULL` or “extern”) to the record header.  

### `rec_1_get_field_start_offs()`, `rec_2_get_field_start_offs()`
- **Purpose**: The “start offset” of field `n` is basically the “end offset” of field `n-1`. These helper functions read from the record header.  

### `rec_get_nth_field_size()` / `rec_get_nth_field_offs()`
- **Purpose**: For old style, we can compute the size or offset of the nth field by analyzing the offset array in the record header.  
- **Use**: This is how we know how many bytes of actual data the nth field occupies on the page (which can be zero if it’s `SQL NULL`).

---

## 6. For New-Style (Compact) Records

Although many of these inline functions are used by old-style code, some are used for new style as well. For instance:
- `rec_offs_comp(offsets)` → checks if a record is in compact format.  
- `rec_get_info_and_status_bits()` → gets both the “info bits” and the “status bits” for a compact record.  

The difference is that in **compact** format, we do not store the field offsets in the record header in the same way. We rely on the `rec_get_offsets()` logic in `rem0rec.c` to parse them. Still, these inline helpers handle the “status bits,” “delete mark,” “heap no.,” “n_owned,” and “next record pointer” for compact records.

---

## 7. “Offsets” Array Helpers

### `rec_offs_base(offsets)`, `rec_offs_n_fields(offsets)`, etc.
- **Purpose**: When we call `rec_get_offsets(rec, index, ...)`, it returns an array that includes the “extra size,” “number of fields,” and then each field’s end offset. These inline macros or functions interpret that array.  
- **Example**: 
  - `rec_offs_n_fields(offsets)` → number of fields in that array.  
  - `rec_offs_any_extern(offsets)` → whether any field is externally stored.  
  - `rec_offs_nth_sql_null(offsets, i)` → if field i is `NULL`.

### `rec_offs_data_size(offsets)` → returns how many bytes the record data occupies.  
### `rec_offs_extra_size(offsets)` → returns how many bytes are used by the “header + offsets” portion.  
### `rec_offs_size(offsets)` → total record size (header + data).  

**Relation to page**: This array is typically used in conjunction with the “physical” record on the page to parse or build the record. The offsets array is allocated in memory (not on the page), but it’s derived from the on-page bits.

---

## 8. Miscellaneous

- `rec_get_data_size_old()` → sums up the sizes of all old-style fields.  
- `rec_fold()` → folds the first `n_fields` and `n_bytes` of a record into a hash value. Used in certain hashing or internal checks.

---

## Summary

1. **Record Headers**: Each record on an InnoDB page has a small header area that indicates “next record offset,” how many fields, whether it’s old/new style, etc.  
2. **Bit-Field Manipulation**: These inline functions carefully read and write the bits in the **record header**. The difference between old and new style mostly involves the layout of these bits (offset from the origin, how many bytes, etc.).  
3. **Offsets Array**: For new-style “compact” records, we often parse the record fields into a memory-based “offsets array.” The inline helpers manipulate that array or the record itself.  
4. **On the Page**: All these offsets are within a single InnoDB page. The “next record pointer,” for instance, is an offset in [0..page_size), so we can jump around within the page.  
5. **In Practice**: Higher-level code like `btr0cur.c` or `row0ins.c` will use these inline functions to navigate records in a page, set the delete-mark bit, link to the next record, or interpret how many fields a record has.

That’s how **`rem0rec.ic`** provides the **low-level** building blocks for managing records *in place* on an InnoDB page.