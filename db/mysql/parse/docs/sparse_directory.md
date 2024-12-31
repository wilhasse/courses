Below is a **concise** overview of **owner records**, **n_owned**, and the **sparse directory** in uncompressed InnoDB pages, along with how they aid in searching rows *within* a single page.

---

## 1. The Sparse Directory: Overview

1. **Location**  
   At the *bottom* (trailer) of each uncompressed InnoDB page is a list of **2-byte slots** known as the **page directory**.

2. **Sparse**  
   - Not every row appears here.  
   - Only a **subset** of records (the “owner” records) get a slot.  
   - The rest of the records are “unowned” and do **not** have a slot.

3. **Why Sparse?**  
   Fewer directory slots means **less overhead** at the bottom of the page. InnoDB still achieves fast lookups because it can do **binary search** on the small list of owners, then follow a short chain of records within the selected owner group.

---

## 2. Owner Records and `n_owned`

1. **Owner Record**  
   - A normal user row (or sometimes infimum/supremum) that appears in the sparse directory.  
   - It has a small integer field, `n_owned`, stored in its record header.

2. **`n_owned`**  
   - Tells how many consecutive records in the singly linked list (including itself) this owner “owns.”  
   - Example: If `n_owned = 4`, that owner covers itself plus the next 3 rows in `REC_NEXT` order.

3. **Groups of Rows**  
   - Because each owner record can “own” some number of subsequent records, InnoDB effectively *clusters* consecutive rows into small groups.  
   - Only the first record (the “owner”) in each group appears in the directory. The others do not.

4. **Dynamic Assignment**  
   - InnoDB decides which records become owners during inserts or page reorganizations.  
   - This is purely an internal heuristic to keep `n_owned` in a reasonable range so no single group is too large or too small.

---

## 3. Searching Within a Page

When InnoDB needs to find a row inside a single uncompressed page, it uses:

1. **Binary Search in the Directory**  
   - The directory slots (owner records) are sorted by their key.  
   - InnoDB compares the search key to these owner records (using their offsets in the page).  
   - Once it finds the slot whose key is just below or equal to the target, it jumps directly to that owner record.

2. **Traverse the Small Group**  
   - From the owner record, InnoDB checks `n_owned` to see how many consecutive rows follow.  
   - It follows `REC_NEXT` pointers (a short chain) until it finds the exact row or passes it.

### Example
Suppose a page has 12 rows (#1 to #12). InnoDB might pick 3 owners, each with its own slot:

- **Owner** #1: `n_owned=3` (covers #1, #2, #3)  
- **Owner** #4: `n_owned=2` (covers #4, #5)  
- **Owner** #6: `n_owned=4` (covers #6, #7, #8, #9)  
- (other rows might belong to #9 or #6’s group, etc.)

If you search for “row #8,” the directory binary search identifies #6’s slot (the last slot < 8). Then you read `n_owned=4` for #6, and follow its short linked chain #6 → #7 → #8 to find the target.

---

## 4. Key Points

1. **This is Intra-Page Only**  
   - The “owner record” concept lives purely inside **one** page.  
   - It has nothing to do with B+‑Tree levels (that’s an inter-page structure).  
   - Once the B+‑Tree logic has located the correct page, InnoDB uses this sparse directory to find (or insert) the exact row on that page quickly.

2. **Space vs. Speed Trade‐Off**  
   - If every record had a slot, the directory would be large.  
   - If no records had a slot, you’d do a linear scan.  
   - By grouping rows under a small number of owners, InnoDB keeps the directory small yet provides faster *in-page* lookups than a purely linear search.

3. **No Direct User Control**  
   - You cannot set “n_owned=10” or pick which records are owners.  
   - InnoDB automatically assigns owners as rows are added or reorganized.

---

## 5. Summary

- **Sparse Directory**: A small list of owner record offsets at the end of each uncompressed page.  
- **Owner Record**: A normal row that appears in the directory and has an `n_owned` field indicating how many subsequent rows it covers.  
- **Fast In-Page Search**: InnoDB can do a **binary search** on the owners, then follow a **short** chain of rows via `REC_NEXT`, dramatically reducing overhead while scanning inside the page.  

These details let InnoDB handle large tables with efficient lookups at both the **tree level** (few page levels) *and* the **page level** (sparse directory).