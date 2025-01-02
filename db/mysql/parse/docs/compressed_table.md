Studying how innochecksum checks compressed page:

First, it’s important to clarify how InnoDB “compressed tables” actually store pages on disk and how innochecksum (and InnoDB’s own code) decides whether to treat a page as “physically compressed.”

## 1. How innochecksum decides if a page is compressed

In the innochecksum code you pasted, **the crucial check** is done in two places:

1. **Reading FSP flags** from the tablespace’s first page (the FSP header at offset `FSP_HEADER_OFFSET + FSP_SPACE_FLAGS`). This sets up a `page_size_t` object that marks whether the tablespace is compressed (`page_size_t(...).is_compressed()`).

2. **If `page_size.is_compressed()`,** innochecksum calls `page_decompress()` on each page (`page_decompress(byte *buf, byte *scratch, page_size_t page_size)`), which eventually calls `os_file_decompress_page()` or a routine like `page_zip_decompress_low()` to decompress that page.

Hence innochecksum basically says: 
> “If the tablespace flags in the first page claim this is a compressed tablespace, then *all* data pages in that .ibd *should* be compressed, so we call `page_decompress()` every time.”

### 1.1 Checking the `FSP_SPACE_FLAGS`
In InnoDB, a “row-format=COMPRESSED” table sets bits inside `FSP_SPACE_FLAGS` that indicate a *physical* page size smaller than the “logical” 16 KB. For example, a KEY_BLOCK_SIZE=8 means an 8 KB physical page, but logically it’s still 16 KB. The core logic in innochecksum is:

```cpp
ulint ssize = FSP_FLAGS_GET_ZIP_SSIZE(flags);
if (ssize > 0) {
    // means compressed
    page_size_t(...).set_is_compressed(true);
}
```

If `ssize == 0`, it’s an uncompressed (or “regular/Antelope/Barracuda but not row-format=COMPRESSED”) file.

That is how innochecksum “identifies” a compressed page: **not** by looking at individual pages to see if they are compressed, but by the *entire tablespace’s* FSP flags.

---

## 2. Are some pages in a “compressed” .ibd still uncompressed?

Yes, in a few corner cases. For example:

1. **Doublewrite buffer pages** are special. Innochecksum either ignores them or does not attempt to verify checksums the same way (the code checks for “if page is doublewrite buffer, skip rewriting checksums”).
2. **FSP header page** (page 0) typically is stored in physically compressed form too if the entire tablespace is compressed, but it is smaller or simpler. In practice, the very first page can still store the FSP header in a “compressed block.” However, InnoDB can sometimes special-case the header.  
3. **Single or partial pages** (like partial last page) might appear uncompressed if the file was truncated or if the tablespace is partially corrupt.  

But in general, if the file’s flags say “compressed = 8K,” *most* pages on disk are physically compressed. The entire .ibd or .ibdata is intended to be consistently physically compressed or not. (In other words, you typically won’t see page 1 compressed but page 2 uncompressed in the same single-table .ibd, except for system / doublewrite pages or certain corner corruption scenarios.)

---

## 3. If you want to conditionally decompress only if the page is “really compressed”

You asked:

> Could I pull code from innochecksum to check if the page is truly compressed, so that if not, I read 16K and skip?

Yes, but typically you do **two** checks:

1. **Check** `FSP_SPACE_FLAGS` in page 0 to see if the entire tablespace is marked compressed. If `FSP_FLAGS_GET_ZIP_SSIZE(...)` is nonzero, you treat *all normal data pages* as compressed.  
2. For any corner cases (like doublewrite pages, or if the table is half truncated), you can attempt a “sanity check” of the decompression. If `page_zip_decompress_low()` fails for a given page, you might treat that page as uncompressed or skip it.  

In practice, InnoDB expects that if the table is declared compressed, then *all pages after page 0* are physically compressed. So it usually does not guess page-by-page. Instead, it does something like:

```cpp
if (page_size.is_compressed()) {
    bool ok = page_zip_decompress_low(...);
    if (!ok) { 
        // page or file might be corrupt
        // or it's special like doublewrite 
    }
} else {
    // normal uncompressed read
}
```

## 4. Could a “compressed” .ibd actually have uncompressed pages?

- **Doublewrite pages** or pages used for certain system overhead can look partially uncompressed or are simply not validated the same.  
- **Partial writes or corruption** might cause a page to appear “invalid compressed data.”  
- In a *multi-file system tablespace* (like `ibdata1` + others), it might have “segments” with different row formats, but it’s unusual.  

So yes, you might see innochecksum do checks to skip or handle special pages, but the fundamental approach is “the entire .ibd is declared compressed or not via the flags in page 0.”

---

## 5. Putting it all together in your code

1. **Read page 0**.  
   - Look at `fsp_header_get_flags()`.  
   - If `FSP_FLAGS_GET_ZIP_SSIZE(flags) > 0`, set `page_size.is_compressed(true)`.

2. **For each subsequent page**:  
   - If `page_size.is_compressed() == true`, attempt to decompress.  
     - If decompress fails, either treat it as corrupt or check if it’s a special page (like doublewrite).  
   - If `page_size.is_compressed() == false`, read as uncompressed.

3. If you want to adopt innochecksum’s “**just skip doublewrite pages**,” you can do something like:

   ```cpp
   bool is_system_ts = (space_id == 0);
   if (is_system_ts && page_no is in [FSP_EXTENT_SIZE..FSP_EXTENT_SIZE*3]) {
       // skip or handle doublewrite buffer region
   }
   ```

4. **No partial guesses**. Innochecksum does **not** do, “Oh, let's try 8K, if that fails let's guess 16K.” It strictly uses the flags from page 0. That’s simpler: if it says “8K compressed,” we do 8K. If it’s not matching in practice, the file is corrupt or belongs to a different table format.

---

### Short Answer

- **Yes**, you can adapt innochecksum’s approach:
  1. Check `FSP_SPACE_FLAGS` on page 0.
  2. If it indicates compression (some `ZIP_SSIZE != 0`), treat *all normal pages* as physically compressed. 
  3. If decompress fails for certain pages, that either means corruption or they’re special (like the doublewrite buffer).  

- **In InnoDB,** once a table is declared “compressed” in the FSP header, all “data pages” really are stored compressed on disk, so you do not pick-and-choose page-by-page. The only exceptions typically are special pages (doublewrite, partial pages, etc.). That’s exactly how innochecksum does it. 

Hence you do *not* normally “read 16K for some pages and 8K for others” if the flags say 8K. The entire single-table .ibd will consistently be 8K on disk. If you read a “16K” chunk from it, you’ll just get half real data + half garbage.  

So, pulling from innochecksum is a good approach—**the code that checks `FSP_SPACE_FLAGS`** is the key to deciding if the *whole* file is physically compressed, and then you apply `page_zip_decompress_low()` to each normal page.