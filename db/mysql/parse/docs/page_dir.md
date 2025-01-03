# Files: zipdecompress.cc and page0zip.cc

zipdecompress.cc - important function: page_zip_decompress_low , definition in the header file.

The file **`zipdecompress.cc`** focuses on the lower-level routines to *decompress* an InnoDB page (the function `page_zip_decompress_low()` and related). Meanwhile, **`page0zip.cc`** is the main interface for both **compressing** and **decompressing** B‑tree pages. It includes `page_zip_compress()` (which uses zlib to compress the page) and calls into `zipdecompress.cc` to perform the actual decompression steps.

### 1. `zipdecompress.cc`: Focused on Decompression
- **Implements**:  
  - `page_zip_decompress_low()`, which does the line-by-line (or record-by-record) logic to rebuild the in-memory B‑tree record format from a compressed page buffer.
- **Used By**:  
  - `page0zip.cc` (and possibly other code) when a compressed page must be decompressed. 
- **Reasoning**:  
  - Keeping decompression in a separate file can be convenient for external tools or utilities that only need to *read* compressed pages, without needing the full InnoDB library or compression code.

### 2. `page0zip.cc`: Compress + Decompress + Support
- **Has**:  
  - `page_zip_compress()`, which uses zlib to generate a compressed representation of an in-memory B‑tree page.  
  - `page_zip_decompress()`, which *calls* `page_zip_decompress_low()` from `zipdecompress.cc`.  
  - Additional helper logic for compressed pages (dense directories, slot management, BLOB pointers, log records, etc.).  
- **Reasoning**:  
  - Provides the top-level interface for the InnoDB *compressed page* system (both compression and decompression, plus page reorganization, logs, etc.).

Hence, the main difference is:  
1. **`zipdecompress.cc`** = the specialized *decompression* routines (detailed, record-level).  
2. **`page0zip.cc`** = the *“front-end”* for compression, plus it delegates to `zipdecompress.cc` for the actual decompression steps.

## page0page.cc

Below is an **overview** of `page0page.cc` (sometimes called `page/page0page.cc`) within InnoDB, how it relates to the rest of the InnoDB codebase, and what each part is doing. This file essentially deals with *general operations on B‑tree index pages*: creation, deletion, validation, plus the “directory” structure used for binary searches (the Infimum, Supremum, page directory slots, etc.).

---

## 1. **Context and Purpose**

1. InnoDB stores tables and secondary indexes in **B‑tree pages**.
2. Each B‑tree page has special “infimum” and “supremum” records at the start and end.
3. There's also a **page directory**, with “slots” that point to selected records to speed up searching.
4. `page0page.cc` provides operations to:
   - Create an **empty** B‑tree page (compact or redundant row format).
   - Insert or remove records in that page.
   - Maintain the “directory” (slots that own groups of records).
   - Validate page integrity (basic consistency checks) even without knowledge of the exact schema.

In InnoDB’s code layout, you’ll often see:

- **`page0page.cc`:**  
  Various *non-compress-specific* routines that are used whenever we manipulate B‑tree pages in memory (e.g., create a new page, delete a list of records, set the max transaction ID, etc.).  
- **`page0zip.cc`:**  
  Specialized logic for compressing/decompressing pages (including some B‑tree–specific code).  
- **`zipdecompress.cc`:**  
  The lower-level record-by-record decompression routines used by `page0zip.cc`.

---

## 2. **Important Data Structures**

### 2.1 Infimum & Supremum
- An index page always starts with an **“infimum”** record (the smallest possible record) and ends with a **“supremum”** record (the largest possible).  
- They aren’t real table rows; they help keep B‑tree logic consistent.

### 2.2 Page Directory
- Stores pointers (slots) to some records so we can do approximate binary searches in the page.  
- Each slot “owns” a small group of consecutive records in the singly linked record list.

For example, “Slot #3” might say “I own 5 records starting from here.” So you find the slot quickly by a binary search in the directory, then linearly scan among those ~5 records.

### 2.3 The Free List
- When a record is removed, it can be added to the “free list” in that page (for re-use).  
- `page_delete_rec_list_*()` routines manipulate that free list.

### 2.4 The Row Format
- **Redundant** or **Compact** row format.  
- In “redundant,” `infimum` = offset `PAGE_OLD_INFIMUM`.  
- In “compact,” `infimum` = offset `PAGE_NEW_INFIMUM`.

Hence there are many conditionals like `if (page_is_comp(page)) {...}` vs. `else {...}` for older row format.

---

## 3. **Key Functions and Their Responsibilities**

Below are the primary routines you will notice in `page0page.cc`:

1. ### `page_create()`, `page_create_low()`
   - **Creates** an empty B‑tree page (either “compact” or “redundant” format).
   - Writes the infimum and supremum, sets up the page directory with 2 slots (infimum, supremum).
   - This is called when allocating new pages in an index (e.g., page splits, new root creation, etc.).

2. ### `page_create_zip()`
   - Specialized version for **compressed** indexes. Creates the uncompressed skeleton in memory and then calls `page_zip_compress()`.

3. ### `page_delete_rec_list_start()` / `page_delete_rec_list_end()`
   - **Deletes** a range of records from the “start” up to a certain record or from a certain record to the “end” of the page.  
   - Adjusts the directory and free list accordingly.

4. ### `page_move_rec_list_*()`
   - Moves records from one page to another, used in page splits, merges, or reorganizing data.  
   - E.g., `page_move_rec_list_end()` is for taking “the last half” of a page’s records and placing them in a new page.

5. ### `page_simple_validate_old()` and `page_simple_validate_new()`
   - Basic consistency checks for old-style (redundant) or new-style (compact) row formats.  
   - Ensures no overlapping records in memory, the infimum/supremum are in the right place, etc.
   - If any big mismatch is found, InnoDB logs a corruption warning.

6. ### `page_validate()`
   - A more advanced check that uses knowledge of the actual `dict_index_t` structure, verifying record ordering is correct, among other checks.  
   - For instance, it checks that records are sorted (in ascending order) as required by the B‑tree, and that each record offset is valid.

7. ### Directory Slot Helpers (`page_dir_*()`)
   - Example: `page_dir_split_slot()`, `page_dir_balance_slot()`, `page_dir_delete_slot()`  
   - Manage how many records a slot “owns,” merges slots, or splits them if they have too many records.

8. ### Others
   - **`page_set_max_trx_id()`**: Sets the `PAGE_MAX_TRX_ID` in the page header (the highest transaction that updated this page).  
   - **`page_update_max_trx_id()`**: Similarly, but possibly with mini-transaction logging.  
   - **`page_rec_print()`, `page_print_list()`, etc.**: Debug printing and diagnostics.  
   - **`page_rec_get_nth()`, `page_rec_get_n_recs_before()`:** Utility for enumerating the singly linked record list and matching it up with directory slots.

---

## 4. **How This Fits in the Bigger Picture**

- **B‑tree operations** (splitting, merging, insertion, deletion) need to manipulate the page’s record list.  
- These top-level operations are typically in files like `btr0btr.cc` or `btr0cur.cc`.  
- Whenever they want to do something to the page (like “move half the records to a new page in a split”), they rely on the **page-level** functions in `page0page.cc`.  
- If the page is in a **compressed** table (`ROW_FORMAT=COMPRESSED`), after we do the in-memory manipulations, InnoDB calls `page0zip.cc` to compress the final result or uses `page_zip_validate()` to confirm correctness.

So, `page0page.cc` is the place where the fundamental layout of the *in-memory B‑tree page* is managed: the *creation* of infimum/supremum, the *linking/unlinking* of records, *validation* that everything is consistent, and so on.

---

## 5. **Summary**

**`page0page.cc`** is all about **managing the fundamental physical layout of B‑tree index pages in memory**:

- **Creating** new pages with infimum and supremum.  
- **Manipulating** or **deleting** segments of records.  
- **Maintaining** the “page directory” to support approximate binary search.  
- **Validating** page integrity and record ordering.  

It does **not** do compression itself (that happens in `page0zip.cc`), but it’s still “page-level” code for InnoDB indexes. This is one of the core parts of how InnoDB organizes data within individual B‑tree pages.

# page0cur.cc

Below is an **overview** of `page0cur.cc` (often called `page/page0cur.cc`) in InnoDB, describing what it does, how it fits in the InnoDB code, and the essential functions within.

---

## 1. **Purpose and Context**

Inside InnoDB, each index page (B‑tree page) can have many “records.” A **page cursor** is a structure that points to a particular record on a page (or to “before the first”/“after the last” positions). This file, `page0cur.cc`, implements logic to:

1. **Search** the correct position of a record within a page using a binary search over the page directory, plus a linear scan among the few consecutive records “owned” by a directory slot.
2. **Insert** a record at the cursor position, if space allows.
3. **Delete** a record at the cursor position.
4. **Update** or move the cursor to the “next” or “previous” record.
5. Provide shortcuts for “adaptive” searching in certain common patterns (like repeated inserts in ascending or descending order).

It also includes specialized logic for:
- **Short inserts** (where the log record is minimized).
- Some help for **R‑tree indexes** (GIS / spatial indexes).
- The possibility of reorganizing or compressing a page if it’s short on space.

---

## 2. **Key Data Structures and Concepts**

1. **`page_cur_t`** (Page Cursor)
   - Tracks the buffer block (`block`) where the page resides and a pointer (`rec`) to the current record.
   - Various functions move this cursor (`page_cur_move_to_next()`, etc.) or position it (`page_cur_set_before_first()`, `page_cur_position()`, etc.).

2. **Directory Slots**
   - The page has a “directory” with slots pointing to certain records. Each slot “owns” a small group of consecutive records for efficient searching.
   - `page_cur_search_with_match()` does a binary search among the directory slots to narrow down to the correct slot, then scans linearly among the small group of records in that slot.

3. **Searching Logic**
   - InnoDB typically does a *binary search* on the directory slots, then a *linear search* in the small group.
   - The code uses `compare()` functions from `dict_index_t` to compare a user’s data tuple (`dtuple_t`) to an on-page record.
   - In addition, “adaptive” logic can skip the binary search step if it thinks the next record is likely adjacent to the last inserted position.

4. **Insertion & Deletion**
   - **Insertion**: Check if enough space is available in the page. Possibly reorganize or compress the page if not. Insert the new record, adjust the singly linked list of records, update the directory slot “owned” count.
   - **Deletion**: Remove the record from the singly linked list, free its space to the page’s free list, update the directory accordingly.

5. **Redo Logging** / Crash Recovery
   - Each of these operations writes minimal redo log records (`MLOG_REC_INSERT`, `MLOG_REC_DELETE`, etc.) so that MySQL can replay them in case of a crash.
   - Some code in `page0cur.cc` specifically handles *parsing* those redo records during recovery (`page_cur_parse_insert_rec()`, etc.).

---

## 3. **Highlighted Functions**

### 3.1 Searching

- **`page_cur_search_with_match()` / `page_cur_search_with_match_bytes()`**  
  Perform the main page-level search for a given data tuple. 
  1. **Binary Search** among directory slots, narrowing to one or two slots that might contain the record.  
  2. **Linear Search** within those few consecutive records.  
  3. Positions the cursor at the record satisfying a given mode (`PAGE_CUR_LE`, `PAGE_CUR_GE`, etc.).

- **Adaptive Shortcut**  
  If the page’s `PAGE_LAST_INSERT` suggests that the new record is close to the last insert position, the code might skip the binary search and attempt to do a local check.

### 3.2 Insertion

- **`page_cur_insert_rec_low()` / `page_cur_insert_rec_zip()`**  
  Insert a record on an **uncompressed** or **compressed** page, respectively. The steps are typically:
  1. Allocate space (check free list or the heap).
  2. Copy the record into the newly allocated space.
  3. Link it into the singly linked list right after the current cursor record.
  4. Update directory slots if needed (splitting or merging “owned” groups).
  5. Possibly reorganize/compress the page if not enough space is found.

- **`page_cur_insert_rec_write_log()`**  
  Write the minimal redo log record for the insertion, capturing only the differences from the “cursor record.”

### 3.3 Deletion

- **`page_cur_delete_rec()`**  
  Deletes a record at the current cursor position:
  1. Removes it from the singly linked list.
  2. Updates “owned” counts in the directory.
  3. Puts the record space into the page’s free list.
  4. Logs the action in the redo log.

### 3.4 Copying Record Lists

- **`page_copy_rec_list_end_to_created_page()`**  
  Specialized function that copies the last half of the records from one page to a newly created page (common in page splits), building up the new page’s infimum & directory. It also logs minimal redo for each inserted record, so crash recovery can replay it.

### 3.5 Redo Log Parsing

- **`page_cur_parse_insert_rec()`**  
  Interprets an `MLOG_REC_INSERT` redo record and re-applies that insert to the uncompressed page (or partial info for compressed page).  
- **`page_cur_parse_delete_rec()`**  
  Interprets an `MLOG_REC_DELETE` redo record and removes that record.

---

## 4. **How `page0cur.cc` Fits In**

- **`btr0cur.cc`** or `btr0btr.cc`**: higher-level B‑tree operations. For instance, b-tree insertion finds the page, then calls `page_cur_insert_rec_*`.
- **`page0zip.cc`**: compression/decompression of entire pages.  
- **`page0page.cc`**: core routines about the page structure, directory, infimum/supremum.  
- **`page0cur.cc`**: specifically about moving around on a page with a “cursor”—search, insert, delete, plus any logging to replay or recover.  
- All these fit together to implement InnoDB’s B‑tree: from searching or splitting pages to writing minimal redo logs for crash recovery.

---

## 5. **Summary**

**`page0cur.cc`** is the main “page cursor” logic within InnoDB, providing:

1. **Efficient Searching** on a single B‑tree page (binary + linear).  
2. **Insertion** next to the cursor (including page reorganization or compression if needed).  
3. **Deletion** of records at the cursor.  
4. **Redo Logging** and **Parsing** for these actions, enabling crash recovery to replay them.  

This layer is crucial for InnoDB’s B‑tree implementation, bridging the “higher-level” b-tree logic and the “low-level” raw page data structures.