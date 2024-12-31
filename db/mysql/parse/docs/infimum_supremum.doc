# Infimum / Supremum

Below is a **concise** yet **in-depth** explanation of **Infimum/Supremum** records in InnoDB pages and **how they relate to B+‑Tree navigation**:

---

## 1. B+‑Tree Overview in InnoDB

InnoDB tables are stored as **B+‑Trees**. Each node in this B+‑Tree corresponds to a **page** (commonly 16 KB). For **non-leaf** pages (internal nodes), InnoDB stores **node-pointer records** that direct queries to child pages. For **leaf** pages, InnoDB stores **user records** containing actual row data.

To navigate from the top of the B+‑Tree to the correct leaf:
1. **Examine node-pointer records** in the current page, decide which child page to follow.
2. **Traverse** until reaching a leaf page that contains user records or is empty (but still has boundary markers).

---

## 2. Infimum and Supremum: “Dummy” Boundary Records

Inside **every** InnoDB page—whether leaf or non-leaf—there are two special, **artificial** records:
1. **Infimum (∞‑lowest)**  
   - Located at a fixed offset (often **99**, labeled `PAGE_NEW_INFIMUM`).  
   - Represents a key “infinitely smaller” than any user record in this page.  
2. **Supremum (∞‑highest)**  
   - Located just after infimum (often **113**, labeled `PAGE_NEW_SUPREMUM`).  
   - Represents a key “infinitely larger” than any user record in this page.

These “dummy” records contain **no user data**. They exist so that internal InnoDB code can always treat the **lowest** or **highest** record boundaries in a consistent way. For example, if a search key is less than every real record on the page, InnoDB effectively compares it to **infimum** to confirm it belongs to the leftmost side. If a key is greater than everything, it is compared to **supremum** to confirm the rightmost side.

### 2.1 They Are Not Compressed
In **compressed** InnoDB pages (ROW_FORMAT=COMPRESSED), the `infimum` and `supremum` bytes are **excluded** from zlib compression. MySQL simply **copies** tiny “infimum_extra” and “supremum_extra” patterns into the uncompressed buffer once inflation completes. Because they are extremely small and always the same, it’s simpler (and more reliable) to keep them outside or trivially reconstructed.

---

## 3. Page Navigation and Infimum/Supremum

In a **non-leaf** (internal) page, to find the correct child page:

1. InnoDB compares your search key against the node-pointer records.  
2. If the key is smaller than all real records, it is effectively greater than the **infimum** but less than the first record—InnoDB goes to the leftmost child.  
3. If the key is bigger than all real records, it is less than **supremum** but bigger than the last user record—InnoDB goes to the rightmost child.

Even on **leaf** pages, `infimum` and `supremum` frame the range of valid user-record keys. A leaf may be empty (no user records), but `infimum` and `supremum` remain present to define boundaries.

---

## 4. Skipping Decompression If You Only Read Header Fields

A **compressed** page usually still has certain **header** fields uncompressed (e.g., `FIL_PAGE_NEXT`, `FIL_PAGE_PREV`, `FIL_PAGE_TYPE`, `FIL_PAGE_LSN`). If you only want to jump from page to page (following the singly/doubly linked list of pages in the index), you can often read those header fields **without** decompressing. 

However, if you want to **read actual user records** or node-pointer records inside the page, you **must** decompress. That decompression is where MySQL reconstructs infimum/supremum in the uncompressed buffer. But those two boundary records themselves are not part of the main zlib stream.

---

## 5. Summary

1. **Infimum & Supremum** exist in **every** InnoDB page to mark “∞‑lowest” and “∞‑highest” boundary keys.  
2. They are **tiny** dummy records that hold no actual user data.  
3. In **compressed** pages, they’re effectively **excluded** from zlib compression, making them straightforward to restore upon inflation.  
4. For **B+‑Tree** navigation (especially searching or inserting), InnoDB always references these boundary records to figure out if a key is less than or greater than all user records in that page.  
5. If you only need header-based page-to-page pointers (e.g., `FIL_PAGE_NEXT`), **no** decompression is required. But if you want to parse real row data or internal node-pointer records, you must **inflate** the compressed region.