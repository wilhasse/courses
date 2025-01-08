# concepts

Below is a simplified explanation of what this piece of InnoDB source code documentation means. It describes how an **index page** is structured internally in InnoDB, specifically how index records are laid out and how they can be searched efficiently.

---

## 1. What Is an Index Page?

Think of a database “page” as the fundamental block of data storage in InnoDB (often 8KB by default). This single page can hold many **index records** (for example, 300 records). This snippet describes how InnoDB organizes these records inside a single page so it can:

1. Quickly look up data.
2. Efficiently insert or delete records.
3. Minimize overhead from splitting, merging, or reorganizing pages.

---

## 2. The Basic Layout

An InnoDB index page contains several parts:

1. **Page Header**  
   At the top, there’s a header containing metadata about the page—its ID, its type (index page), how many records it contains, etc.

2. **A Linear List (Heap) of Records**  
   Below the page header, all the individual index records are stored in a **linear linked list**. They’re sorted in **alphabetical (or ascending key) order** and are “chained” from one to the next.

3. **Page Directory (Array of Pointers)**  
   At the bottom of the page, just before the page end, there is a small array of pointers. Each pointer in this directory points to one record in the linked list. These pointers are used to speed up lookups with a form of “binary search.”  
   
   Typically, **not every record** is pointed to—just about **every 6th (or so)** record. Because the pointers are fewer, we keep a small array that can be binary searched quickly.

---

## 3. How the “Ownership” and “Directory Slots” Work

- Each entry in the page directory points to a specific record in the list.
- The record pointed to by this directory slot “owns” itself and a certain number of subsequent records (in the range 4 to 8).  
- The **4-bit field** in the record marks how many records are “owned” by it (from that record up to, but **not including**, the record owned by the previous directory slot).

The first and second slots in the directory are special:
- Slot **0** (the very first slot) usually owns only **1** record (just itself).
- Slot **1** might own somewhere between **1 and 8** records.
- Subsequent slots typically own **4 to 8** records.

**Why do this?** It is a balancing trick so that if you do a binary search among the directory slots, once you narrow down to the right slot, you don’t have too many records to scan linearly.

### Example

Imagine you have 12 records total, sorted alphabetically. Suppose your page directory looks like this:

```
Directory Slot (I)    Points To Record Key    "Ownership" Count
--------------------------------------------------------------
         0                     A                    1
         1                     C                    4
         2                     H                    4
         3                     N                    3
```

- Slot 0 points to record “A” and owns only itself (count = 1).  
- Slot 1 points to “C” and owns 4 records total (C, D, E, F).  
- Slot 2 points to “H” and owns 4 records (H, I, J, K).  
- Slot 3 points to “N” and owns 3 records (N, O, P).

So if you were looking for “J,” you’d do a binary search among slots `[0..3]`, see it belongs after slot 1 and before slot 3, jump to slot 2 (which points to H), and scan linearly through records H, I, J, K until you find “J.”

---

## 4. Why Is This Helpful for Inserts and Deletions?

### Inserts

- **Most** inserts just append (push) the new record onto the heap (linear list).  
- **Occasionally** (for about every 8th insert), we need to update the directory. Because the directory is relatively small (maybe ~50 pointers for 300 records), the overhead of shifting pointers is small.  

### Deletions

- Deleting a record means removing it from the linked list and updating the **4-bit “owned-records” field** in the “owner” record.  
- If the count of owned records drops too low (or you remove the owner itself), you might also update the page directory.

This approach lets InnoDB handle small changes (inserts/deletes) cheaply, without having to do a full page reorganization every time.

---

## 5. Page Split, Merge, or Reorganization

- If the page becomes **completely full** but still has many “deleted” records (space is not actually reusable because they’re just marked deleted), or if we are **splitting** or **merging** pages (like in a B-tree split), InnoDB may do a deeper **reorganization**.  
- Reorganization means rewriting the list of records more tightly and rebuilding the directory so everything is balanced again.

---

## 6. Why Might We Increase Page Size?

The documentation mentions the possibility of increasing the page size from 8 KB to 64 KB if the disk transfer rate improves relative to seek and latency time. The overhead of inserts/deletes doesn’t grow too large because:
- The main cost to keep track of records is in the small directory array.
- The “heap” plus these partially-filled directory slots remains manageable even if the page can hold many more records.

In other words, if it becomes faster to stream bigger chunks of data (larger page) than to seek multiple small pages, a bigger page can be beneficial.

---

## 7. A Concrete “Numbers” Example

- **Page Size**: 8 KB (8,192 bytes)  
- **Typical Number of Index Records**: ~300  
- **Directory Size**: ~50 pointers, each pointer is 4 bytes → 50 × 4 = 200 bytes  

So out of 8 KB, the directory itself only uses 200 bytes, the header uses some bytes, and the rest is available for the actual records.

---

## Summary in a Nutshell

1. There is a **page header** at the top with metadata.  
2. All **index records** are arranged in **one linear list** in alphabetical/key order.  
3. At the **bottom**, a **page directory** has pointers (in sorted order) to every 6th or so record, allowing **binary search** on the directory.  
4. Each pointer “owns” a small group of consecutive records, controlled by a 4-bit field in the record.  
5. **Insertion** is mostly cheap (just put the record in place); occasionally, the directory is updated.  
6. **Deletion** just updates the list and the ownership count.  
7. **Reorganization** of the page only happens when we split, merge, or have many “stale” deleted records.  

The result: quick lookups with a small directory to binary search in, combined with cheap inserts/deletes for day-to-day operations.

---

**That’s it!** This is the simplified explanation of how an InnoDB index page is structured, how records are stored, and how the directory pointers make searching faster while keeping insertion and deletion overhead low.

# page0page.c

Below is a **step-by-step** walkthrough of selected functions and code excerpts, linking them back to the high-level description of how an **InnoDB index page** is structured (from your previous question). This explanation aims to clarify:

1. **How the page is organized**: Page header, linear list of records, directory array (slots), infimum and supremum records, etc.  
2. **How these functions** interact with or manipulate the data structure: insertion, deletion, heap management, directory balancing, etc.  
3. **Important fields** like `n_owned`, `heap_no`, and the usage of “owner” records.

---

## 1. Key Concepts Recap

Remember the main structures on an InnoDB **index page**:

- **Page Header**: Holds metadata (like `PAGE_N_RECS`, pointers to free space, etc.).  
- **Linear List (Records)**: Infimum record → user records (sorted) → supremum record.  
- **Page Directory** (array of “slots” at the bottom): ~ every 6th record has a pointer in this array, helps with binary-search-like lookups.  
- **Owned Records**: Each slot points to a record, and that record “owns” itself plus some following records (the 4-bit `n_owned` field).  
- **Infimum**: Minimal “fake” record at the start.  
- **Supremum**: Maximum “fake” record at the end.

Most operations revolve around:
- **Inserts**: Usually appended in a linear list, occasionally rebalancing the directory.  
- **Deletes**: Remove from the linear list, update `n_owned` and possibly the directory.  
- **Splits / merges**: Restructure the page when too full or merging.  
- **Checks / Validations**: Ensure integrity (no overlap, correct pointers, etc.).

---

## 2. Example Functions Explained

Below, we detail some of the functions in **page0page.c** that operate on this structure.

### 2.1. `page_dir_find_owner_slot()`
```c
ulint
page_dir_find_owner_slot(const rec_t* rec) { ... }
```
- **Purpose**: Given a record `rec`, find which directory slot “owns” it.  
- **Key Steps**:
  1. The code climbs forward through the record list until it finds a record whose `n_owned` is nonzero. That record is the “owner.”  
  2. It then compares the offset of that owner record with each slot in the directory until it matches. That slot is considered the directory slot for `rec`.
- **Relates to**:  
  - The `n_owned` field: shows how many records are “owned” by this record.  
  - Directory array: each slot points to a “owner record,” so this function sees which slot lines up with that record’s offset.

---

### 2.2. `page_dir_slot_check()`
```c
UNIV_STATIC
ibool
page_dir_slot_check(page_dir_slot_t* slot) { ... }
```
- **Purpose**: Validate a directory slot: check that the record it points to is consistent (e.g., `n_owned` in the correct range, correct referencing).  
- **Relates to**:
  - Ensuring `n_owned` obeys the rule: typically `4 <= n_owned <= 8`, but can be `1` for the first slot, etc.  
  - Confirms slot is within a valid pointer range.

---

### 2.3. `page_set_max_trx_id()`
```c
UNIV_INTERN
void
page_set_max_trx_id(buf_block_t* block, page_zip_des_t* page_zip, trx_id_t trx_id, mtr_t* mtr)
```
- **Purpose**: On each page, InnoDB may store the “highest transaction ID” that modified that page. This helps with MVCC and cleanup.  
- **Relates to**:
  - Not directly about the directory or record layout, but an example of storing page-level metadata in the page header.

---

### 2.4. `page_mem_alloc_heap()`
```c
UNIV_INTERN
byte*
page_mem_alloc_heap(page_t* page, page_zip_des_t* page_zip, ulint need, ulint* heap_no)
{ ... }
```
- **Purpose**: Allocate memory for a **new record** from the “heap top” pointer within the page.  
- **Key Steps**:
  1. Check if enough space is available.  
  2. If so, move `PAGE_HEAP_TOP` downward by `need`.  
  3. Assign the new record a “heap_no” (like an ID in the page’s record-heap).  
- **Relates to**:
  - The page’s “heap” usage. In InnoDB, all record bodies are in a single “heap” region, allocated from top to bottom.  

---

### 2.5. `page_parse_create()`, `page_create()`, and `page_create_low()`
```c
UNIV_INTERN
page_t* page_create(buf_block_t* block, mtr_t* mtr, ulint comp) { ... }

UNIV_STATIC
page_t* page_create_low(buf_block_t* block, ulint comp) { ... }
```
- **Purpose**: Create a brand-new index page with the **infimum** and **supremum** records.  
- **Key Steps** (`page_create_low`):
  1. Initializes the page with a known structure (infimum at `PAGE_NEW_INFIMUM`, supremum at `PAGE_NEW_SUPREMUM`).  
  2. Sets up the directory with **2 slots** (slot 0 → infimum, slot 1 → supremum).  
  3. Sets initial `n_owned = 1` for both infimum and supremum.  
- **Relates to**:
  - The fundamental page structure: always start with two “dummy” records, infimum & supremum.  
  - Sets up initial directory with exactly 2 slots.

---

### 2.6. `page_copy_rec_list_end_no_locks()`
```c
UNIV_INTERN
void
page_copy_rec_list_end_no_locks(buf_block_t* new_block,
                                buf_block_t* block,
                                rec_t* rec,
                                dict_index_t* index,
                                mtr_t* mtr)
{ ... }
```
- **Purpose**: Copy all records from `rec` onward **to another page** (`new_block`) while ignoring locks. This is part of splitting or reorganizing the B-tree.  
- **Key Steps**:
  1. Iterates from `rec` to the end (supremum).  
  2. Inserts each record into `new_block` using a low-level method (`page_cur_insert_rec_low`).  
  3. Helps in a page split: half the records go to the new page.  

---

### 2.7. `page_copy_rec_list_end()` and `page_copy_rec_list_start()`
```c
UNIV_INTERN
rec_t* page_copy_rec_list_end(...)

UNIV_INTERN
rec_t* page_copy_rec_list_start(...)
```
- **Purpose**: Move a portion of records from one page to another, either from `rec` to the end (`page_copy_rec_list_end`) or from the beginning up to `rec` (`page_copy_rec_list_start`).  
- **Relates to**:
  - **Splitting pages**: when a page is too full, half the records move to a new page.  
  - Adjusts `PAGE_MAX_TRX_ID`, the directory pointers, etc.  

---

### 2.8. `page_delete_rec_list_end()` and `page_delete_rec_list_start()`
```c
UNIV_INTERN
void
page_delete_rec_list_end(rec_t* rec, buf_block_t* block, ...)

UNIV_INTERN
void
page_delete_rec_list_start(rec_t* rec, buf_block_t* block, ...)
```
- **Purpose**: Actually remove records from a page’s linear list, from “rec” onward (includes `rec`) or up to (but not including) `rec`.  
- **Key Steps**:
  1. Adjust the “owner record’s” `n_owned` field.  
  2. Possibly fix up the directory array (e.g., the pointer that used to refer to these removed records might be updated).  
  3. Chain the removed portion into the page’s **free list** so the space can be reused.  
- **Relates to**:
  - The idea that “deleting a record” is removing it from the singly-linked chain and updating `n_owned`.  
  - Freed records get linked into a free list for potential reuse.

---

### 2.9. `page_dir_delete_slot()` and `page_dir_add_slot()`
```c
UNIV_INLINE
void
page_dir_delete_slot(page_t* page, page_zip_des_t* page_zip, ulint slot_no)

UNIV_INLINE
void
page_dir_add_slot(page_t* page, page_zip_des_t* page_zip, ulint start)
```
- **Purpose**:  
  - **Delete**: Remove an entire slot from the directory. The records it “owned” then become owned by the next slot.  
  - **Add**: Insert a new directory slot (e.g., if a slot was “owning” too many records, we split it).  
- **Relates to**:
  - The dynamic nature of the directory array. Usually 50–60 pointers for ~300 records, but can change.  
  - The “split slot” scenario when `n_owned` exceeds the maximum threshold (like 8).

---

### 2.10. `page_dir_split_slot()` and `page_dir_balance_slot()`
```c
UNIV_INTERN
void
page_dir_split_slot(page_t* page, page_zip_des_t* page_zip, ulint slot_no)

UNIV_INTERN
void
page_dir_balance_slot(page_t* page, page_zip_des_t* page_zip, ulint slot_no)
```
- **Purpose**:  
  - **split_slot**: If a single slot “owns” too many records, we add a slot in the middle.  
  - **balance_slot**: If a slot owns fewer than the minimum (4 by default), try to move records from an upper neighbor or merge the slots.  
- **Relates to**:
  - Maintaining that each slot typically owns 4–8 records so lookups remain balanced.

---

### 2.11. `page_validate()` (and the simpler `page_simple_validate_old()` / `page_simple_validate_new()`)
```c
UNIV_INTERN
ibool
page_validate(page_t* page, dict_index_t* index)
```
- **Purpose**: Comprehensive integrity check for an index page.  
- **Key Steps**:
  1. Check that the directory doesn’t overlap the record heap.  
  2. Traverse the record list from infimum to supremum:  
     - Ensure each record is well-formed.  
     - Check the `n_owned` fields line up with the directory slots.  
     - Confirm ascending order (if we know the index).  
  3. Check the free list of deleted records.  
  4. Compare the total data size with the sum of each record’s size.  
- **Relates to**:
  - Making sure the entire data structure (linked list + directory) is consistent.  
  - If any pointer or `n_owned` is out of place, “page_validate” might detect the corruption.

---

### 2.12. `page_print()`, `page_print_list()`, `page_dir_print()`, etc.
These are **debugging** or **diagnostic** functions that print out:

- **Page Header** data (like `PAGE_N_RECS`, `PAGE_HEAP_TOP`, etc.).  
- **The directory**: which slots point to which offsets, and how many records they own.  
- **The entire record list**: each record’s key fields, `n_owned`, next pointer, etc.

They are invaluable when debugging or analyzing corruption.

---

## 3. Putting It All Together

1. **Creating a new page**: `page_create_low` sets up infimum and supremum, gives them directory slots 0 and 1, each with `n_owned = 1`.  
2. **Inserting a record**: Usually done by allocating space (`page_mem_alloc_heap`), placing the record in the linked list, adjusting `n_owned` for its “owner.” Every so often, we might add or balance directory slots.  
3. **Deleting a record**: Remove it from the chain (by adjusting `rec_get_next_offs(rec)` on the previous record). If it was “owned” by a slot, decrement that slot’s `n_owned`. If `n_owned` goes below threshold, rebalance. Freed space is appended to `PAGE_FREE`.  
4. **Splitting a page**: We use `page_copy_rec_list_end` or `page_copy_rec_list_start` to move about half of the records to a new page. Then we delete them from the old page. The directory gets updated accordingly.  
5. **Validation**: `page_validate` checks the entire layout for internal consistency.  
6. **Debugging**: `page_print` and `page_dir_print` let you see exactly how records and slots are laid out.

---

## 4. Summary

In essence, these C functions are InnoDB’s internal machinery for **managing B-Tree index pages**. The core ideas of:

1. A **linear list** with infimum & supremum.  
2. A **directory** for quick approximate lookups.  
3. Small overhead for inserts/deletes.  
4. Carefully maintained “ownership” fields (`n_owned`).  

…are implemented by the various helper methods seen above. The code is quite detailed because it must handle **crash recovery, concurrency, and performance** while preserving data consistency in the B-Tree structure.

# page0cur.c

Below is a **function-by-function** explanation of the key parts of **`page0cur.c`** (the “page cursor” file in InnoDB). It references how these routines fit into the **index page** organization (page header, linked list of records, directory array, etc.) that we explained previously.

---

## 1. Overview: What Is a “Page Cursor”?

A **page cursor** is basically a pointer (or iterator) into a single InnoDB index page that allows you to:

1. Position yourself on a specific record in the page’s linked list.
2. Move forward or backward (e.g., `page_cur_move_to_next`).
3. Insert or delete records relative to the current record.
4. Perform partial or complete **searches** on a single page.

Whereas higher-level “B-tree code” (like `btr0cur.c`) navigates across multiple pages, **page0cur** is the lower-level mechanism to traverse or manipulate records **within** a single page.

---

## 2. Key Data Structures Recap

- **Page Header** (`page_header_...`): Has the `PAGE_N_RECS`, `PAGE_LAST_INSERT`, `PAGE_HEAP_TOP`, etc.  
- **Records**: Infimum → user records (in key order) → supremum, each with:
  - A pointer to the next record in the singly-linked list.
  - Possibly some metadata: `n_owned`, `heap_no`, etc.
- **Page Directory** (`page_dir_slot_t` array): Pointers to some “owner records” for binary-like search.  
- **Cursor** (`page_cur_t`): 
  - `cursor->rec`: points to the “current record.”  
  - `cursor->page`: references the page memory.  
  - Low-level methods to re-link records, adjust ownership, etc.

---

## 3. Detailed Look at Selected Functions

### 3.1. **`page_cur_try_search_shortcut()`**  
```c
UNIV_INLINE
ibool
page_cur_try_search_shortcut(
    const buf_block_t* block,
    const dict_index_t* index,
    const dtuple_t* tuple,
    ...
    page_cur_t* cursor)
```
- **Purpose**: A “shortcut” optimization. If we’ve just recently inserted data into the **last inserted** record (`PAGE_LAST_INSERT`), we can sometimes guess that the new search is near that record, so we skip a full binary search.  
- **How It Works**:
  1. Grab `PAGE_LAST_INSERT` from the page header.
  2. Compare the `tuple` we’re searching for with the last inserted record and its successor record.
  3. If the new value falls between them, we position the cursor there and say “success = TRUE.” Otherwise, no shortcut.  
- **Bigger Picture**: This is an InnoDB performance trick. Page-level “adaptation” tries to cut down the cost of repeated inserts or lookups in a consecutive range.

---

### 3.2. **`page_cur_search_with_match()`**  
```c
UNIV_INTERN
void
page_cur_search_with_match(
    const buf_block_t* block,
    const dict_index_t* index,
    const dtuple_t* tuple,
    ulint mode,
    ulint* iup_matched_fields,
    ...
    page_cur_t* cursor)
```
- **Purpose**: Perform a **binary search** on the page directory to find where `tuple` should go (or to locate `tuple`) in the record list.  
- **Key Steps**:
  1. **Binary search** among the directory slots to narrow down a “low” slot and “high” slot.  
  2. Then **linear search** the chain of records between those two slots.  
  3. Once we find the correct position, we set `cursor->rec` to that record.  
  4. The `mode` can be `PAGE_CUR_L` (strict less), `PAGE_CUR_LE` (less or equal), `PAGE_CUR_G` (strict greater), or `PAGE_CUR_GE` (greater or equal), so we decide exactly how to position.  
- **Note** the code around `iup_matched_fields` and `iup_matched_bytes`:  
  - This is an optimization to keep track of how many fields (and partial bytes) we already matched to avoid re-computing from scratch.  

---

### 3.3. **`page_cur_open_on_rnd_user_rec()`**  
```c
UNIV_INTERN
void
page_cur_open_on_rnd_user_rec(
    buf_block_t* block,
    page_cur_t* cursor)
```
- **Purpose**: Positions the cursor on a **random** user record in the page.  
- **Key Steps**:
  1. Get the total number of user records (`page_get_n_recs`).  
  2. Generate a random number in `[0, n_recs)`.  
  3. Move from the before-first record that many steps forward.  
- **Why**: Useful for certain testing or sampling scenarios within a single page.

---

### 3.4. **Inserting a Record with `page_cur_insert_rec_low()`**
```c
UNIV_INTERN
rec_t*
page_cur_insert_rec_low(
    rec_t* current_rec,
    dict_index_t* index,
    const rec_t* rec,
    ulint* offsets,
    mtr_t* mtr)
```
- **Purpose**: **Insert** a new record **after** `current_rec`. This is for **uncompressed** pages.  
- **Key Steps**:
  1. Compute the record size.  
  2. See if we can reuse a “free” record spot (see if there’s anything in `PAGE_FREE` list). If not, allocate from the `PAGE_HEAP_TOP`.  
  3. Copy the new record data into that newly allocated area.  
  4. **Link** it into the singly linked list: the “next” pointer of `current_rec` is changed to our new record, and the “next” pointer of the new record is set to what used to be `current_rec`’s next.  
  5. **Increment** `PAGE_N_RECS`.  
  6. **Update** the “owner” record’s `n_owned`, possibly splitting a directory slot if we exceed the limit.  
  7. Write a redo log record about this insert (`page_cur_insert_rec_write_log`).  
- **Relation to Directory**: If the “owner record” had 7 records and we just inserted an 8th, we might do `page_dir_split_slot(...)` to keep each slot owning 4–8 records.

#### Compressed Page Variant: `page_cur_insert_rec_zip()`
- Almost identical, but handles the **compressed page** specifics (there is a `page_zip` descriptor, and we also must keep compressed data consistent).

---

### 3.5. **Deleting a Record: `page_cur_delete_rec()`**
```c
UNIV_INTERN
void
page_cur_delete_rec(
    page_cur_t* cursor,
    dict_index_t* index,
    const ulint* offsets,
    mtr_t* mtr)
```
- **Purpose**: Deletes the record at `cursor->rec`.  
- **Key Steps**:
  1. **Log** the delete in the redo log (`page_cur_delete_rec_write_log`).  
  2. **Unlink** it: find the record’s predecessor in the singly linked list, then point that predecessor’s “next” to the record’s “next.”  
  3. Fix up directory ownership (`n_owned`) if the “owner record” is losing this record. Possibly call `page_dir_balance_slot(...)` if we go below the minimum threshold.  
  4. Release the space into the page’s “free list.”  
  5. The cursor is repositioned to the **next** record after the deleted one.  

---

### 3.6. **`page_copy_rec_list_end_to_created_page()`**
```c
UNIV_INTERN
void
page_copy_rec_list_end_to_created_page(
    page_t* new_page,
    rec_t* rec,
    dict_index_t* index,
    mtr_t* mtr)
```
- **Purpose**: Copies **all** records from `rec` to the supremum into a **newly created** page. Used in **page splits** or other reorganizations.  
- **Key Steps**:
  1. Iterate over the records from `rec` until the supremum.  
  2. For each record, copy it into `new_page`, building up the singly linked list (infimum → new rec → new rec → ... → supremum).  
  3. Construct new directory slots as needed (in this snippet, it tries to mimic the logic used by incremental inserts so that recovery matches normal operations).  
  4. Write redo logs for each inserted record in “short insert” mode.  

---

### 3.7. **Redo Log Parsing Helpers: `page_cur_parse_insert_rec()`, `page_cur_parse_delete_rec()`, etc.**
When InnoDB recovers from a crash, it replays “insert” and “delete” log records.  
- **`page_cur_parse_insert_rec()`**: Rebuilds or re-applies an **insert** by reading the log record.  
- **`page_cur_parse_delete_rec()`**: Same for a **delete**.  
- They piece together the correct offset within the page, figure out how many bytes changed, and call the normal insert/delete routines so the page structure is consistent again.

---

### 3.8. **Random Utility: `page_cur_lcg_prng()`**
```c
UNIV_STATIC
ib_uint64_t
page_cur_lcg_prng(void)
```
- This is just a small linear congruential generator (LCG) random number generator used by `page_cur_open_on_rnd_user_rec()`.  
- It seeds from `ut_time_us(NULL)`. Each call returns a 64-bit pseudo-random number.  
- Not directly about the page structure, but helpful in test features (pick random record).

---

## 4. Putting It All Together

- **Searching**: You have code like `page_cur_search_with_match()` to do partial binary/linear searches within the page’s directory+list.  
- **Inserting**: `page_cur_insert_rec_low()` or `page_cur_insert_rec_zip()` handles the actual memory allocation, linking, and directory balancing.  
- **Deleting**: `page_cur_delete_rec()` unlinks a record, updates ownership, and adds the space to the free list.  
- **Page Splits**: Often use “copy record list end to new page,” then “delete” them from the old page.  

These “page0cur.c” functions wrap the lower-level “page0page.c” primitives (like setting `n_owned`, updating directory slots, etc.). Together, they manage how we step through records, do a quick local search, or do local inserts/deletions in an InnoDB index page.

# page0zip.c

Below is a **function-by-function** (or logical section) walkthrough of **`page0zip.c`**, which handles **compressed page** operations in InnoDB. It links back to the ideas of **index pages**, **records**, and the **page directory** as discussed previously, but now with **zlib-based compression** concepts and “dense directory” details.

---

## 1. Why a “Compressed Page” in InnoDB?

When **InnoDB** is configured to use a **compressed** page format, each 8KB (or other size) page is internally compressed. Instead of storing the entire uncompressed data directly on disk, InnoDB holds:

1. **An uncompressed page** in memory (the usual structure with page header, linear list of records, page directory, etc.).  
2. **A compressed copy** (`page_zip_des_t`), which is the zlib-compressed data plus additional metadata for partial in-memory operations.

Hence, you’ll see code that **deflates** (compresses) or **inflates** (decompresses) portions of the page. This `page0zip.c` file manages that process, including:

- **Dense Directory**: A compressed data structure listing all user records (and their offsets) in sorted order, separate from infimum/supremum.  
- **Recompression** or **Reorganization**: If a page is updated (records inserted/deleted) and becomes too large, InnoDB tries to “recompress” it. If that fails, it may reorganize the page in memory, then compress again.  
- **Apply Log**: If a page has partially updated data, a small “modification log” helps reconstruct the final page on decompression.

---

## 2. Key Data Structures and Concepts

1. **`page_zip_des_t`**: A descriptor for the compressed page. 
   - `data`: Pointer to the raw compressed bytes.  
   - `ssize`: The compressed page size.  
   - `m_start`, `m_end`, `m_nonempty`: Indicate the boundaries of the small “modification log” appended to the compressed stream.  
   - `n_blobs`: Tracks the number of externally stored columns (BLOB pointers) in this page.

2. **Dense Directory**: Instead of a “page directory” with ~ every 6th record, in the compressed representation we store **every** user record’s offset in a small array at the “end” of the compressed buffer. This array is “dense” (one entry per user record).  

3. **Infimum and Supremum**: In the uncompressed page, we always store these “dummy” records. But in the **compressed** form, they are omitted from the zlib-compressed data. We re-inject them on decompression.

4. **Zlib**: Standard library used for compression. We set up a `z_stream` and call `deflate()` or `inflate()` with some custom memory allocator.

5. **Modification Log**: When a record is updated or inserted, we don’t always want to recompress the entire page. Instead, we append a tiny redo-like “log entry” of the changes to the tail of the compressed data. On decompression, we reconstruct the final page by reading these log entries.

---

## 3. Major Functions in `page0zip.c`

### 3.1. **page_zip_compress()**
```c
ibool
page_zip_compress(
    page_zip_des_t* page_zip,
    const page_t*   page,
    dict_index_t*   index,
    mtr_t*          mtr)
```
- **Purpose**: Compress an **uncompressed** InnoDB page (`page`) into the `page_zip->data` buffer using zlib.  
- **Steps**:
  1. **Check** the uncompressed page structure (e.g., infimum and supremum must be correct).  
  2. **Build** a “dense directory” that lists each user record offset.  
  3. **Deflate** (zlib) the page contents, skipping infimum/supremum, storing only user records.  
  4. For special columns (like `DB_TRX_ID` or externally stored BLOB references), store them uncompressed at the tail.  
  5. If successful, write a log record (`MLOG_ZIP_PAGE_COMPRESS`) so recovery can replay it.  
  6. On failure, `FALSE` is returned but the old compressed data is left intact.

### 3.2. **page_zip_decompress()**
```c
ibool
page_zip_decompress(
    page_zip_des_t* page_zip,
    page_t*         page,
    ibool           all)
```
- **Purpose**: Decompress a **compressed** page from `page_zip->data` into a normal InnoDB page (`page`).  
- **Steps**:
  1. Reconstruct the infimum and supremum in the uncompressed page.  
  2. “Inflate” (zlib) the stored bytes.  
  3. Rebuild the singly linked list of records.  
  4. Apply any “modification log” entries appended at the end of the compressed buffer.  
  5. On success, `page` is now a fully valid uncompressed page. On failure, returns `FALSE`.

### 3.3. **Dense Directory Operations** (e.g., `page_zip_dir_encode()`, `page_zip_dir_decode()`, `page_zip_dir_insert()`, `page_zip_dir_delete()`)
- **Purpose**:  
  - `page_zip_dir_encode()` constructs the dense directory from the uncompressed page’s records in ascending order.  
  - `page_zip_dir_decode()` rebuilds the uncompressed page’s “sparse” directory from the dense list, hooking up records.  
  - `page_zip_dir_insert()` / `page_zip_dir_delete()`: Adjust the dense directory after an insert or delete (shifting offsets, etc.).

### 3.4. **BLOB Pointers and “Externally Stored Columns”**  
- Some columns (BLOB, TEXT, etc.) might be “externally stored.” In compressed pages, the references (`BTR_EXTERN_FIELD_REF`) are stored uncompressed in a special area.  
- Functions like `page_zip_write_blob_ptr()` or code in `page_zip_compress_clust_ext()` handle copying or skipping these pointers in/out of the compressed buffer.

### 3.5. **page_zip_reorganize()**
```c
ibool
page_zip_reorganize(
    buf_block_t* block,
    dict_index_t* index,
    mtr_t* mtr)
```
- **Purpose**: If simply calling `page_zip_compress()` fails, we can “reorganize” the page (like a mini defragmentation), then try compress again.  
- **What Happens**:
  1. Temporarily copy out all records.  
  2. Re-create the page fresh (new infimum/supremum, empty space).  
  3. Copy the records in a tight layout.  
  4. Attempt `page_zip_compress()` again.  
  5. If that still fails, we return `FALSE`.

### 3.6. **Small Utility Routines**  
- **`page_zip_set_extra_bytes() / page_zip_set_owned() / page_zip_rec_set_deleted()`**: Mark whether a record is “owned” or “deleted” in the compressed structure.  
- **`page_zip_dir_find()`**: Locate a record in the dense directory.  
- **`page_zip_write_header_log()`, `page_zip_calc_checksum()`, `page_zip_parse_compress()`**: Parsing or writing log entries related to compression, checksums, etc.  

---

## 4. How Does This Relate to the (Un)Compressed Page Layout?

- **In the normal uncompressed page**: We have a page header (metadata), a singly linked list of infimum → user records → supremum, and a sparse directory of ~ every 6th “owner.”  
- **In the compressed form**:
  - We leave out the infimum/supremum for zlib.  
  - We gather **every** user record offset in a contiguous “dense” array at the end.  
  - The actual record bytes (minus externally stored columns) get deflated by zlib.  
  - Some special fields (like `DB_TRX_ID`) might be placed uncompressed in a small reserved region.

Hence, the entire “page0zip.c” code is about bridging these two representations safely, applying partial updates, and ensuring we can re-inflate the compressed data into a valid InnoDB page.

---

## 5. Summary

1. **page0zip.c** is the “plumbing” that handles how InnoDB’s compressed pages are structured and updated.  
2. It uses **zlib** to compress all user records except infimum/supremum.  
3. It also maintains a **dense directory** and a small “modification log” appended at the end of the compressed buffer.  
4. On decompression, it reconstructs the infimum, supremum, the singly linked user records, and any log-based changes.  
5. If compression fails due to insufficient space, InnoDB attempts a **page reorganization** to better pack the data.  

Overall, this code ensures that InnoDB can store index pages in a smaller, compressed footprint on disk, while still allowing partial in-memory modifications without having to fully re-zip everything on every small change.