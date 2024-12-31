# Why is so difficult do decompress Innodb pages?

**Short Answer**  
Yes, the table really is compressed, but **InnoDB’s compression format** is more than just a plain zlib “inflate from PAGE_DATA onward.” The small code snippet you’ve been using (which simply calls `inflate()` from offset 38/`PAGE_DATA`) will often produce invalid offsets (like `57444`) because **InnoDB’s compressed pages** have additional metadata and structures (the modification log, directory arrays, BLOB references, etc.) embedded in the payload.

If you only call naive `inflate()`, you will end up with **garbled** uncompressed pages and thus out-of-bounds pointers.

---

## Why Your Current Approach Yields Bad Offsets

When you do:
```cpp
// naive pseudo-code
memcpy(page_out, compressed_page, PAGE_DATA); // copy first 38 bytes
inflate(..., compressed_page + 38, page_out + 38, ...);
```
…the resulting `page_out` is **not** a correctly reconstructed InnoDB page. It might look mostly correct at first glance, but the **“dense directory,” “modification log,”** and possibly parts of the B-tree header aren’t set up as the real MySQL server would. As soon as you parse “infimum” or read `REC_NEXT`, you get nonsense offsets like 57444, far outside 16 KB.

---

## InnoDB’s Compression Format Isn’t Just “zlib at Byte 38”

Inside the MySQL codebase, **`page0zip.cc`** (and `page0zip.h`) handle many extra steps:

1. A **dense directory** is appended at the end of the “compressed stream.”  
2. The **modification log** is placed after the compressed data portion (the so-called “m_nonempty,” “m_end,” etc.).  
3. For **clustered indexes** (ROW_FORMAT=COMPRESSED), there may be **BLOB references** (externally stored columns) that get partially stored out-of-line.  
4. Some fields (like `FIL_PAGE_PREV`, `FIL_PAGE_NEXT`, `FIL_PAGE_LSN`) remain partially uncompressed in the header. The code re-checks them after inflation.

So the MySQL server does something like:

1. **Copy** certain page header fields from the uncompressed buffer.  
2. **Inflate** only the parts that truly store record data.  
3. **Manually** fix up (or skip) the “modification log” space.  
4. **Manually** reconstruct the page directory.  
5. Potentially store some fields (like DB_TRX_ID, DB_ROLL_PTR) uncompressed at the end.  

**Naive inflation** doesn’t do these extra steps—so your final “uncompressed page” is not a valid InnoDB page structure.

---

## How to Read Real Compressed Pages

### 1) Use the Full `page_zip` Logic from MySQL
If you look at [MySQL’s `page0zip.cc`](https://github.com/mysql/mysql-server/blob/8.0/storage/innobase/page/page0zip.cc) (or the 5.7 equivalent), you’ll see a function like:
```c
bool page_zip_decompress(page_zip_des_t* page_zip, page_t* page, bool all);
```
It’s about 600–900 lines of code. It:

- Reads the compressed page header.  
- Allocates z_stream, sets up the correct boundaries.  
- Inflates *only* the record data region.  
- Rebuilds the directory from the “dense directory.”  
- Handles BLOB pointer references if needed.

If you **truly** want the correct, uncompressed 16 KB InnoDB page, you have to replicate *all* that logic. 

### 2) Use an Existing Utility (like `innochecksum` or `page_parser`)  
MySQL ships some debugging tools that can parse compressed pages. For example:

- **`innochecksum --page-type`** (in MySQL 5.7+ it can do some basic checks, though not a full record parse).  
- The InnoDB “page_parser” utility in `storage/innobase/page` can parse compressed pages for debugging.  

### 3) Use an External Library (e.g. “innodb_ruby,” “innodb-java-reader,” etc.)  
Some open-source projects have partially re-implemented InnoDB’s page compression logic in a higher-level language. If you only need to **inspect** or **export** records, this can be simpler than rewriting MySQL’s entire code.

### 4) Switch Table to ROW_FORMAT=DYNAMIC or COMPACT  
If your real goal is just to read `.ibd` files offline, you can avoid the complexity of “InnoDB compression” by not using ROW_FORMAT=COMPRESSED. If that’s an option, you can re-create or alter the table to `ROW_FORMAT=DYNAMIC`. Then the pages are not zlib-compressed at rest, and your parser can handle them easily.

---

## Conclusion

You are indeed opening a **compressed** table, which explains why your “offset from previous record” is huge. **InnoDB’s compressed page format** is **not** just “zlib from byte 38.” You need the additional logic in MySQL’s `page0zip.cc` to produce a valid uncompressed page data structure.

- If you need a **quick solution**: re-import the table with `ROW_FORMAT=DYNAMIC` or `ROW_FORMAT=COMPACT`, so your custom parser can keep working.  
- If you *must* parse compressed pages: replicate the entire official MySQL “page_zip” logic, or use an existing library that fully supports InnoDB compression.  

That’s why you’re seeing out-of-range offsets. It’s expected with a naive inflation approach on a real ROW_FORMAT=COMPRESSED `.ibd`.


# Decompress page (draft)

Below is a **minimal** C-only snippet that demonstrates how InnoDB’s compressed page “payload” can be **inflated (decompressed)** using zlib.  
This snippet is extracted and slightly simplified from MySQL’s `page0zip.cc` code. It is meant as a “proof of concept” rather than a drop-in replacement, because the real MySQL code references many internal macros, structures, and functions.

> **Disclaimer**  
> - This example will **not** compile out of the box against the MySQL code base without further adaptation.  
> - You still need the definitions of page sizes, offsets, etc. If you have MySQL source available, you can integrate these definitions from `include/page0zip.h`, `include/page0types.h`, etc.  
> - The snippet omits many asserts (`ut_ad()`, `ASSERT_ZERO()`), debug macros, and integrity checks.  
> - This is only the *decompress* logic—i.e. “inflate.” It will build a fresh uncompressed `page[]` from the compressed content in `page_zip->data`.  

Nevertheless, it should give you a starting point to **test if you can decompress a page**. 

---

# Minimal `page_zip_decompress.c`

```c
/***************************************************************************
 * The main function that the MySQL code calls to decompress a page.
 * In MySQL 8, this logic is split into page_zip_decompress() and
 * page_zip_decompress_low().
 * 
 * This snippet is purely demonstrative. It will NOT compile by itself.
 ***************************************************************************/

/** Decompress a page.
 @return true on success, false on failure */
bool page_zip_decompress(
    page_zip_des_t *page_zip, /*!< in/out: compressed page descriptor:
                               data[] (the raw compressed block),
                               ssize (size in bytes, e.g. 16384) */
    page_t *page,             /*!< out: uncompressed page (size=page_zip->ssize) */
    bool all)                 /*!< in: if true, copy certain page header fields
                               that normally remain unchanged after page creation */
{
    // This function measures time for stats, then calls:
    return page_zip_decompress_low(page_zip, page, all);
}

/** The heavy-lifting function for decompressing an InnoDB page.
 @return true on success, false on failure */
bool page_zip_decompress_low(
    page_zip_des_t *page_zip, /*!< in/out: compressed page descriptor */
    page_t *page,             /*!< out: uncompressed 16KB (or 8KB) page */
    bool all)                 /*!< in: see above */
{
    // The MySQL code references many helpers:
    //  - z_stream d_stream
    //  - page_zip_dir_decode()
    //  - page_zip_set_extra_bytes()
    //  - page_zip_decompress_{clust,node_ptrs,sec}()
    //  - page_zip_fields_decode()
    //  - mem_heap_t, dict_index_t, etc.

    z_stream d_stream;
    dict_index_t *index = nullptr;
    rec_t **recs;
    ulint n_dense;    // number of user records (excluding infimum/supremum)
    ulint trx_id_col; // position of DB_TRX_ID, used if it's a clustered index
    mem_heap_t *heap;
    ulint *offsets;

    // Basic checks
    if (!page_zip || !page) {
        return false;
    }
    // e.g. validate pointer alignment, etc.

    // 1) The "dense directory" size check
    n_dense = page_dir_get_n_heap(page_zip->data) - PAGE_HEAP_NO_USER_LOW;
    if (n_dense * PAGE_ZIP_DIR_SLOT_SIZE >= page_zip_get_size(page_zip)) {
        // Something is off if the directory doesn't fit
        return false;
    }

    // 2) Create a working heap
    //   (In MySQL, mem_heap_create() is a custom memory manager.)
    heap = mem_heap_create(...);

    // 3) Allocate array for the "record" pointers
    recs = static_cast<rec_t**>(mem_heap_alloc(heap, n_dense * sizeof(*recs)));

    // 4) Possibly copy or verify the page header
    if (all) {
        // If "all == true", copy the entire 38-byte or so header from
        // the compressed page to the uncompressed page.
        memcpy(page, page_zip->data, PAGE_DATA);
    } else {
        // If "all == false", only copy certain fields that can change
        // after page creation. MySQL 8 does some checks to ensure that
        // the rest of the page header matches what we already have in `page`.
        // For brevity, we skip details.
    }

    // 5) Decode the "dense directory" from the compressed page
    //    and place real record offsets in `recs[]`.
    //    If it fails (e.g. inconsistent offsets), return false.
    if (!page_zip_dir_decode(page_zip, page, recs, n_dense)) {
        mem_heap_free(heap);
        return false;
    }

    // 6) Reconstruct infimum & supremum records in `page[]`
    //    (In InnoDB, these are stored uncompressed or trivially.)
    memcpy(page + (PAGE_NEW_INFIMUM - REC_N_NEW_EXTRA_BYTES),
           infimum_extra, sizeof infimum_extra);
    // If page is empty, set next to PAGE_NEW_SUPREMUM, else link it to
    // the first user record, etc.
    // Similarly, copy the supremum bytes.

    // 7) Initialize z_stream
    memset(&d_stream, 0, sizeof(d_stream));
    // MySQL sets up a custom "page_zip_set_alloc()" for z_stream allocators
    if (inflateInit2(&d_stream, UNIV_PAGE_SIZE_SHIFT) != Z_OK) {
        // Some kind of zlib init error
        mem_heap_free(heap);
        return false;
    }

    // 8) Prepare the z_stream input to skip the uncompressed header
    d_stream.next_in  = page_zip->data + PAGE_DATA;
    d_stream.avail_in = static_cast<uInt>(page_zip->ssize - (PAGE_DATA + 1));
    // The +1 is for the "end marker" of the modif log

    // 9) Prepare the z_stream output to fill from PAGE_ZIP_START onward
    d_stream.next_out  = page + PAGE_ZIP_START;
    d_stream.avail_out = (ulint)(page_zip->ssize - PAGE_ZIP_START);

    // 10) Perform a partial inflate to decode the zlib header
    //     and the "index field info." MySQL does multiple calls to inflate():
    //     e.g. inflate(..., Z_BLOCK), inflate(..., Z_BLOCK)
    int zerr = inflate(&d_stream, Z_BLOCK);
    if (zerr != Z_OK && zerr != Z_BUF_ERROR) {
        inflateEnd(&d_stream);
        mem_heap_free(heap);
        return false;
    }

    // MySQL does a second inflate(..., Z_BLOCK) to decode the dictionary info
    zerr = inflate(&d_stream, Z_BLOCK);
    if (zerr != Z_OK && zerr != Z_BUF_ERROR) {
        inflateEnd(&d_stream);
        mem_heap_free(heap);
        return false;
    }

    // 11) Now parse the "index field info" from the uncompressed
    //     portion in `page + PAGE_ZIP_START` vs d_stream.next_out.
    //     That yields a dict_index_t describing # of columns, fixed-len vs var-len,
    //     and possibly sets trx_id_col if it's a clustered index.
    index = page_zip_fields_decode(
        page + PAGE_ZIP_START,
        d_stream.next_out,
        page_is_leaf(page) ? &trx_id_col : nullptr,
        (fil_page_get_type(page) == FIL_PAGE_RTREE /* bool is_rtree? */));
    if (!index) {
        inflateEnd(&d_stream);
        mem_heap_free(heap);
        return false;
    }

    // 12) Now do the actual record inflation, depending on node-pointer vs leaf
    d_stream.next_out = page + PAGE_ZIP_START; // reset

    // Pre-allocate "offsets" array for rec_get_offsets_reverse()
    // MySQL uses 1 + 1 + dict_index_get_n_fields(index) + REC_OFFS_HEADER_SIZE, etc.
    offsets = static_cast<ulint*>(mem_heap_alloc(heap, (some_size) * sizeof(ulint)));

    if (!page_is_leaf(page)) {
        // This is a node-pointer page
        if (!page_zip_decompress_node_ptrs(
                page_zip, &d_stream, recs, n_dense, index, offsets, heap)) {
            goto err_exit;
        }

        // page_zip_set_extra_bytes() sets the "info bits" for infimum if FIL_PAGE_PREV == FIL_NULL
        // etc.
        if (!page_zip_set_extra_bytes(page_zip, page, 
               (mach_read_from_4(page + FIL_PAGE_PREV) == FIL_NULL) 
                   ? REC_INFO_MIN_REC_FLAG : 0)) {
            goto err_exit;
        }
    } else {
        // It's a leaf. Check if it’s a clustered index
        if (trx_id_col == ULINT_UNDEFINED) {
            // secondary index
            if (!page_zip_decompress_sec(page_zip, &d_stream, recs,
                                         n_dense, index, offsets)) {
                goto err_exit;
            }
        } else {
            // clustered index
            if (!page_zip_decompress_clust(page_zip, &d_stream, recs,
                                           n_dense, index, trx_id_col,
                                           offsets, heap)) {
                goto err_exit;
            }
        }

        if (!page_zip_set_extra_bytes(page_zip, page, 0)) {
            goto err_exit;
        }
    }

    // 13) Done inflating. We have a valid page with a correct dense directory,
    //     record data, infimum/supremum, node pointers, etc.
    inflateEnd(&d_stream);

    // free index descriptor, or store it
    page_zip_fields_free(index);

    // free temporary heap
    mem_heap_free(heap);

    return true;

err_exit:
    inflateEnd(&d_stream);
    page_zip_fields_free(index);
    mem_heap_free(heap);
    return false;
}
```

## 3. Minimal Steps if You Want to “Pull In” MySQL’s `zipdecompress.cc`

**If** you still want to integrate all relevant code from MySQL 8.0 “`zipdecompress.cc`,” you must:

1. **Grab**:
   - `zipdecompress.cc` (which you already have partial code from).  
   - `zipdecompress.h`  
   - All headers it includes: `btr0btr.h`, `mem0mem.h`, `page0zip.h`, `page0page.h`, `rem0rec.h`, `dict0dict.h`, `rem0wrec.h`, …  
2. Provide **stubs** or **real** definitions for any references to `srv_cmp_per_index_enabled`, `page_zip_stat[]`, `mutex_enter()`, `mutex_exit()`, `MONITOR_INC()`, `buf_LRU_stat_inc_unzip()`, etc.  
3. Provide **all** the macros/consts: `PAGE_ZIP_DIR_SLOT_SIZE`, `PAGE_ZIP_START`, `UNIV_PAGE_SIZE_SHIFT`, `ULINT_UNDEFINED`, etc., typically in `page0size.h`, `page0types.h`, `univ.i`, etc.  
4. Provide **the entire** `dict_index_t` definition, or at least enough of it so that calls like `dict_index_get_n_fields(index)` compile. That usually means pulling in a chunk of `dict0dict.cc`, plus “table” objects, etc.  
5. Provide a `mem_heap_t` implementation from `mem0mem.cc`.

You end up bundling a **lot** of InnoDB code. You also need to compile them with the same macros MySQL uses (`-DUNIV_LINUX`, `-DHAVE_CONFIG_H`, etc.) to ensure you pick up the right ifdefs.  

**In short**, you must basically embed large parts of InnoDB. This is why MySQL themselves keep it inside the server, not as a small standalone library.

