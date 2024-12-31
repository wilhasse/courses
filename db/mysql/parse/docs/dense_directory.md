# Directories

In InnoDB’s on‐disk page format, there is a **page directory** at the *end* (the “trailer”) of each page that helps locate records efficiently. However, the **exact structure** of this directory differs between **uncompressed** pages and **compressed** pages. In the InnoDB code, you will frequently see references to the terms:

1. **Sparse directory** (on **uncompressed** pages)  
2. **Dense directory** (on **compressed** pages)

Below is an in-depth look at **what** they are, **why** both exist, and **how** they are used.

---

## 1. The Uncompressed “Sparse Directory”

### 1.1 Sparse Directory Basics
On an **uncompressed** InnoDB page (ROW_FORMAT=Compact or Dynamic), InnoDB stores a list of **2‐byte offsets** (sometimes called “slots”) at the *very end* of the 16KB (or 8KB, etc.) page. This list is known as the **“page directory.”**

- Each slot corresponds to the **“start”** (or sometimes “end”) of a group of user records.
- Why “sparse”? Because **not** every record has its own slot. Instead, **one record** in each “group” (the record that *owns* the group) has a slot. That owner’s `n_owned` field indicates how many records in that group follow it.

Hence, the slots in the uncompressed directory are **fewer** than the total number of records, making it “sparse.” For instance, if you have 100 records in the page, they might be grouped into ~10 owners. That means only ~10 directory entries, rather than 100.

### 1.2 Infimum & Supremum
Even in uncompressed pages, InnoDB keeps the “infimum” record at offset 99 (PAGE_NEW_INFIMUM) and “supremum” at offset ~113. Typically those do **not** appear in the page directory, because they are “artificial boundary records.”

### 1.3 Why Use This “Sparse” Approach?
It saves space. For large pages with many records, storing a slot for **every** record can be wasteful. InnoDB’s “grouping” approach means you only store a directory slot for “owner” records, and each owner’s `n_owned` field indicates how many records are in that group. Thus, fewer directory entries at the end.

---

## 2. The Compressed “Dense Directory”

### 2.1 The Problem with Compression
When InnoDB compresses a page (ROW_FORMAT=COMPRESSED), the standard “sparse directory” alone is **not** enough to reconstruct everything after inflation. Among other issues:

- You might have records on the free list (deleted rows).
- Records can move around or partially update, requiring a “modification log.”

Thus, MySQL internally decided it’s simpler to store a “**dense directory**” in the compressed page: it has **an entry for every record**, including those on the free list, plus infimum & supremum references. When you decompress, you can walk this dense directory to build a standard “sparse” page directory again.

### 2.2 Dense Directory Contents
In a compressed page:

1. **Every user record** (including “deleted” ones) gets a 2‐byte entry describing its offset and flags (delete‐mark, owned‐flag, etc.).  
2. These entries live in the “trailer” of the compressed page, right after the zlib‐inflated region.  

Hence, “dense” means there is a slot **for every record**, not just owners. That’s more data than the uncompressed (sparse) approach but makes it easier to reconstruct the **exact** set of record offsets once you do the inflation.

### 2.3 “Decoding” the Dense Directory
After you run zlib inflation on the record data, InnoDB calls a function like `page_zip_dir_decode(...)` to read these dense entries. It sorts them by their offsets, checks which ones are “owned,” and then constructs the normal “sparse directory” in the last portion of the uncompressed 16KB page. Only then does the page look like a standard InnoDB page with `n_owned` groups, etc. That’s why the compression code calls it “**decode**.”

In short, the **dense directory** is a **temporary representation** in the compressed page format. Once decompressed, it is transformed into the usual “sparse directory” structure in the uncompressed page.

---

## 3. Putting It All Together

1. **Uncompressed Page (Sparse Dir)**  
   - Physically at the bottom of the 16KB page.  
   - Contains **fewer** slots than the number of records—just one slot per “owner.”  
   - Freed (deleted) records do **not** necessarily appear in the directory, because they’re on a free list.

2. **Compressed Page (Dense Dir)**  
   - Physically near the end of the *compressed* data region (the “trailer”).  
   - Contains **one slot for every record**, including deleted ones.  
   - Requires a decode step (`page_zip_dir_decode()`) to build the normal sparse directory after inflation.

3. **Why Not Just Keep Sparse?**  
   - When row data is compressed, we also need to keep track of every record offset (including free‐list) to do partial updates, handle BLOB references, etc.  
   - A “dense” listing is simpler to manipulate at compression time. On decompression, MySQL can methodically rebuild the normal “uncompressed” structures.

---

## 4. Example Flow in Decompression

1. **We have a 16KB “compressed page”** in memory. At the end of that data is a series of 2‐byte entries for **every** record—**the dense directory**.  
2. **We inflate** the record data portion with zlib.  
3. **We call `page_zip_dir_decode()`,** which reads those 2‐byte dense entries. They are sorted by offset, filtered for “delete‐marked” vs. “owned,” and used to build the standard sparse directory in the last ~2*N_OWNED bytes of the now‐uncompressed 16KB page.  
4. We also reconstruct infimum/supremum, plus the free list of deleted records, so that the final page matches the normal “uncompressed InnoDB page layout.”  
5. **Now** the page can be navigated by normal InnoDB code: `rec_get_next_offs()` reads `REC_NEXT` from each record, the sparse directory, etc.

---

## 5. Key Takeaways

- **Sparse Directory** = uncompressed format’s directory at the bottom of the page, storing only “owner” records.  
- **Dense Directory** = compressed format’s listing of **every** record offset (including free list).  
- During decompression, MySQL uses the dense directory to re‐build a proper sparse directory.  

If you do not handle the dense directory logic, you cannot properly reconstruct the uncompressed page’s offsets—and you’ll see random or out‐of‐range offsets (like 50,000+) when reading the “infimum” record’s `REC_NEXT`.  

Hence, **both** the “sparse” and “dense” directories are part of InnoDB’s **page format**. The “sparse” approach is used in normal uncompressed pages to reduce overhead, while the “dense” list in compressed pages ensures MySQL can fully reconstruct the page.