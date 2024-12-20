# btr0btr.c

**Overview**

In early versions of InnoDB (such as the Embedded InnoDB 1.0.6 codebase), table data and secondary indexes are stored in a data structure known as a B+‑tree. The source file **btr0btr.c** (or btr0btr.cc in some distributions) is part of the low-level B-tree layer of InnoDB. This code, along with related files like `btr0cur.c`, `btr0pcur.c`, and `btr0sea.c`, provides the foundational routines for managing InnoDB’s B+‑tree indexes. These routines implement essential operations such as locating pages, splitting nodes, merging nodes, inserting and deleting records, navigating the tree, and maintaining the structural integrity of the index.

When discussing "early InnoDB," we refer to the implementation details from the time before InnoDB became fully integrated into MySQL (especially prior to MySQL 5.6 and its subsequent improvements), and before the code was extensively refactored. At this stage, the code organization and naming conventions were somewhat more direct and less abstracted than in modern InnoDB versions.

------

**Key Concepts in Early InnoDB B-Tree Implementation**

1. **B+‑Tree Structure**:
   InnoDB indexes are implemented as B+‑trees:
   - **Non-leaf (internal) pages** contain "node pointers" and "separator keys" that guide the search down the tree.
   - **Leaf pages** store actual records or pointers to row data. Clustered indexes (the primary index on a table) store the full row data in the leaf pages, while secondary indexes store index entries plus a reference (the primary key or a hidden clustered key) to the actual row data.
2. **Page Format**:
   Each page in InnoDB’s data files is 16 KB by default. B-tree pages have a carefully structured format:
   - A **page header** that stores metadata (page type, number of records, page-level locks, etc.).
   - **User records** (or index entries) arranged in a doubly-linked list.
   - **Page directory** slots that help in binary searching within the page.
   - **Fil header/trailer**, which includes checksums, LSNs (log sequence numbers), and pointers for linking pages into a file space structure.
3. **Record Format**:
   Records in leaf pages contain the actual data fields or key fields for secondary indexes. There is a "record header" that includes information about the record’s length, the "next record" pointer, and various flags. The B-tree code frequently deals with navigating to the correct record, determining if a new record fits, and understanding how much space is available within a page.
4. **Multi-Version Concurrency Control (MVCC)**:
   Although not fully contained in `btr0btr.c`, the B-tree code interacts closely with the transaction system and MVCC. Records often include hidden system columns (like DB_ROW_ID, DB_TRX_ID, DB_ROLL_PTR) that help manage concurrency and rollback. The B-tree code ensures that when records are inserted or modified, they are done so in a way consistent with MVCC rules.

------

**The Role of btr0btr.c**

`btr0btr.c` (and similarly named files) primarily contain “core” B-tree routines. While `btr0cur.c` manages cursor operations (navigating through a B-tree, positioning to a specific key), and `btr0pcur.c` deals with persistent cursors (used for certain sustained scans or operations), `btr0btr.c` is often the home for more fundamental functions that:

- **Create and Initialize B-Trees**:
  Functions to create a new index page or an entire B-tree root. This includes formatting the page headers and directories, and inserting initial boundary ("infimum" and "supremum") records that define the logical start and end of the page’s record list.
- **Splitting and Merging Pages**:
  When a page is full and a new record must be inserted, the tree must split. Similarly, if records are removed and pages become too sparse, the tree might merge pages. `btr0btr.c` contains logic to handle page reorganizations:
  - **Page Splits**: Determine where to split, allocate a new page, redistribute records, and adjust pointers upward in the tree.
  - **Page Merges**: Identify when a sibling page can absorb records from a nearly empty page and remove the empty page from the structure.
- **Balancing the Tree**:
  While B-trees are generally “balanced” by their nature, operations that insert or delete records can cause localized imbalances. The code ensures that the depth of the tree remains consistent and that no single leaf or internal page becomes overloaded or underloaded relative to siblings.
- **Allocating and Freeing Pages**:
  InnoDB uses a space management system inside data files. `btr0btr.c` routines may interact with `fsp` (file space) or `buf` (buffer) modules to allocate new pages for splits and to mark pages as free when merging or deleting nodes.
- **Navigational Aids**:
  While most navigation logic is in `btr0cur.c`, some underlying routines for searching down the tree from the root, following child pointers, and comparing keys at internal nodes appear in `btr0btr.c`. This code uses binary search techniques within each page’s directory array to locate the correct child pointer or record.
- **Consistency Checks and Debugging**:
  Early InnoDB code includes assertions and debug builds that verify tree consistency. `btr0btr.c` often includes checks that ensure page structures are valid, tree pointers align correctly, and no logical contradictions occur between pages.

------

**Important Data Structures and Functions**

- **`btr_leaf_split` / `btr_page_split` (or similarly named functions)**:
  Handle the complexity of splitting a page that is full. They:
  - Choose a "split point" based on record sizes and desired load distribution.
  - Allocate a new sibling page.
  - Move half of the records to the new page.
  - Update parent nodes if necessary (if the page being split is not the root).
- **`btr_root_create`**:
  Creates a root page when an index is first built. This involves initializing the page headers and placing the minimal boundary records (infimum and supremum) that make the page structurally complete even when empty.
- **`btr_page_delete` / `btr_merge_pages`**:
  Inverse of page splits. When a page becomes too empty, these functions try to merge it with a sibling. If successful, one page is freed and the parent node pointers are updated accordingly.
- **Record Insertion and Deletion Hooks**:
  While the actual insertion of a record into a leaf is often done via functions in `btr0cur.c`, `btr0btr.c` provides lower-level support and verification routines. It might handle preparing space on a page or adjusting record pointers as the tree structure changes.

------

**Interaction With Other Modules**

- **Buffer Pool (`buf0buf.c`)**:
  The B-tree code must fetch pages into memory (the buffer pool) before reading or modifying them. `btr0btr.c` interacts with the buffer manager to "fix" a page in memory and then "unfix" it after operations.
- **Logging and Recovery (`log0log.c`, `trx0trx.c`)**:
  Changes to the B-tree structure are logged for crash recovery. Each page split, merge, or record insert/deletion is accompanied by writing “redo log” entries. While `btr0btr.c` does not implement logging itself, it ensures that logging hooks are called to maintain transactional consistency.
- **Dictionary and Metadata (`dict0dict.c`)**:
  The B-tree code operates on pages and indexes identified by internal structures that come from the dictionary cache. These structures provide essential metadata about the index layout, key structure, and compression settings. `btr0btr.c` uses this information to interpret records and keys correctly.

------

**Evolution Over Time**

The early InnoDB code for B-trees was already quite sophisticated but less modular and more interwoven than modern code. Over subsequent versions, the InnoDB team refactored and reorganized:

- The code became clearer, with more descriptive function names.
- More robust error handling and diagnostic instrumentation were added.
- Better separation of responsibilities allowed easier maintenance and feature additions (like online DDL, advanced compression, and atomic DDL operations).

However, the core logic—InnoDB using a B+‑tree for indexes, relying on careful page-level operations, and maintaining balance and integrity—remains largely the same. `btr0btr.c` from the early InnoDB era is the ancestor of these refined operations, serving as the foundational set of routines for building and maintaining B-trees in the InnoDB storage engine.

------

**Conclusion**

In summary, `btr0btr.c` in early InnoDB is a central building block in the implementation of B+‑trees. It provides low-level mechanisms to create, split, merge, and maintain B-tree pages and ensure that indexes remain balanced and consistent. Understanding this file and its related modules (`btr0cur.c`, `btr0pcur.c`, `btr0sea.c`) gives insight into how InnoDB structured its on-disk data and managed indexes, enabling efficient data retrieval, insertion, deletion, and transactional integrity in MySQL databases.

# btr0cur.c

`btr0cur.c` is part of the B-tree layer in early InnoDB. It implements the operations related to B-tree cursors, including:

- Navigating in a B-tree: searching for a specific key, moving to next/previous record
- Inserting and updating records through a cursor
- Deleting or delete-marking records
- Handling externally stored (BLOB) fields

This file interacts closely with the low-level page operations (`page0page.c`), space management (`fsp0fsp.c`), logging (`mtr0mtr.c`), and record format utilities (`rem0rec.c`, `rem0cmp.c`). In early InnoDB code, many of these operations are tightly coupled and less abstracted than in newer code.

## Key Concepts and Structures

- **B-tree Cursor (btr_cur_t)**:
  A `btr_cur_t` object represents a position in a B-tree. It holds references to the current page and record, and sometimes path information used for estimates or certain searches. The cursor abstracts the details of navigating between records and pages.
- **Mini-Transactions (mtr_t)**:
  The code makes heavy use of mini-transactions (`mtr_t`). These are lightweight transactional units for page-level changes, ensuring consistent, atomic modifications at the page level. MTR logging calls like `mlog_write_ulint` or `mlog_open()` appear frequently, writing redo-log records as changes are applied.
- **Compressed and Uncompressed Pages**:
  Even in early InnoDB, code accounts for compressed pages (`WITH_ZIP`). Some functions handle logic differently depending on whether a page is compressed or not. By MySQL 5.7 and 8.0, page compression support and related code paths are better modularized and documented.
- **Externally Stored Fields (BLOBs)**:
  Large columns may be stored outside the B-tree page. The code includes specialized logic for reading, writing, freeing, and updating these external fields (BLOB parts). This is quite intricate in early code, but still foundational for how InnoDB manages large objects.

## Main Functionalities in the Snippet

### Searching and Positioning Cursors

- **`btr_cur_search_to_nth_level()`**:
  Conducts a binary search down the B-tree to position the cursor at a given level. It uses node pointer pages for navigation, descends down the tree, and finally latches the leaf page as requested.
  In newer code (MySQL 5.7/8.0), search routines are still conceptually similar, but code is cleaner, more modular, and better commented. There are more helper functions, and adaptive hash index (AHI) lookups and page prefetching are integrated with clearer abstractions. Error handling and latching rules are more explicit.
- **`btr_cur_open_at_index_side_func()` and `btr_cur_open_at_rnd_pos_func()`**:
  These position the cursor at the start/end of the index or at a random position. They illustrate how InnoDB can open a cursor without a particular key—useful for range scans.
  In modern InnoDB, the logic remains similar, but the code is cleaner, often with separate functions and better naming conventions. Also, improvements in read-ahead, buffering, and concurrency make these operations more robust and easier to follow.

### Insert, Update, and Delete Operations

- **Inserts (`btr_cur_optimistic_insert()` and `btr_cur_pessimistic_insert()`)**:
  The code tries an optimistic insert first (no page split) and if that fails due to lack of space, it tries a pessimistic insert (page splits allowed, more logging). It also handles writing undo logs, acquiring locks, and possibly storing big fields externally.
  Modern versions still have optimistic and pessimistic paths, but code is more structured. There are clearer distinctions between logical (locking, transaction) and physical (page splitting, record format) steps. Also, improved error reporting and code comments make it easier to understand what’s happening.
- **Updates (`btr_cur_update_in_place()`, `btr_cur_optimistic_update()`, `btr_cur_pessimistic_update()`)**:
  Updating a record involves lock checks, undo logging, and ensuring there is enough space on the page (or performing reorganizations or splits if not). If the record size changes, the code tries to handle it optimistically first. If that fails, it falls back to a pessimistic approach involving page splits or reorganizations.
  Newer versions encapsulate these complexities more cleanly. Changes in record size, handling of BLOBs, and undo logging have more modular code and often separate utility functions. The logic for page reorganizations and merges is now clearer, with better factored-out code.
- **Delete Marking and Deleting (`btr_cur_del_mark_set_clust_rec()`, `btr_cur_del_mark_set_sec_rec()`, `btr_cur_optimistic_delete()`, `btr_cur_pessimistic_delete()`)**:
  The code includes logic to “delete-mark” a record first (logical deletion) and actually remove the record later, possibly triggering page compression or even page merges. Handling delete marks involves writing redo logs, updating transaction fields, and taking care of locks. Physically removing a record may require further space management steps.
  Modern code still follows the same high-level steps, but is more robust and better commented. The logic for when to compress, when to merge pages, and how to handle logging and rollback is more straightforward and better tested.

### Externally Stored Fields (BLOBs)

Much of the latter part of the snippet deals with externally stored fields (BLOB columns):

- **`btr_store_big_rec_extern_fields()`**:
  Writes the large externally stored columns to separate pages. This involves allocating new pages for the BLOB parts, writing out their data, and updating the record’s field reference.
  In newer code, the handling of large columns is more isolated. Functions to read/write BLOB parts, handle compressed/uncompressed BLOB pages, and manage their life cycle are better documented and use safer coding patterns.
- **`btr_free_externally_stored_field()` and related functions**:
  These free BLOB pages when a record that owns them is deleted or rolled back. The code walks through the chain of BLOB pages, freeing each in turn.
  In newer versions, you’ll find more robust error handling and possibly more consistent naming and modularization of these routines. Also, better integration with the redo/undo logs makes recovery more reliable.

### Logging and Recovery Integration

The snippet shows frequent calls to `mlog_write_ulint()`, `mlog_open()` etc., which are used to write redo log records. The code tries to ensure that every structural change is logged. Modern versions keep this pattern but with clearer abstractions. MySQL 8.0 code often includes more systematic logging steps and better handles edge cases.

## Summary

The provided code snippet from `btr0cur.c` in Embedded InnoDB 1.0.6.6750 is a dense, low-level implementation of B-tree cursor operations, including searching, inserting, updating, deleting, and handling big externally stored fields. Compared to MySQL 5.7 and 8.0:

- The basic logic and data structures remain recognizable.
- The newer code is more modular, clearer, and more thoroughly documented.
- Modern InnoDB code has improved naming conventions, richer instrumentation, better error handling, and improved performance/concurrency features.
- Externally stored column (BLOB) handling, page compression, and logging are more robust and easier to follow in later versions.

In essence, the early InnoDB code in `btr0cur.c` shows the foundation of how InnoDB manages B-tree operations. Newer versions build on these foundations, refining, restructuring, and clarifying the code to make it easier to maintain, understand, and enhance.

# btr0pcur.c

The provided code snippet is from **btr0pcur.c**, a part of InnoDB’s B-tree layer in earlier versions of InnoDB. This file deals with **persistent cursors** (pcur) that remain stable across operations and can be used to efficiently resume navigation of a B-tree after certain operations or transaction boundaries. Below is a thorough explanation of what this code does, its significance, and how it relates to the broader InnoDB architecture, as well as how it compares to more modern InnoDB code.

### What is a Persistent Cursor?

A **persistent cursor** in InnoDB is an abstraction that allows a B-tree position to be stored and restored without holding a latch on the page indefinitely. For example, consider a scenario where a transaction scans through an index or needs to remember its position while it performs other operations. Instead of holding a page latch (which would prevent other transactions from making progress), the code uses a persistent cursor to:

1. Store the current position in the B-tree (including key prefix information).
2. Release page latches to allow others to access the page.
3. Later restore the cursor position (re-latch the appropriate page and move the cursor to the correct record).

This mechanism helps improve concurrency by minimizing the time pages remain latched, which is crucial in multi-threaded database environments.

### Key Operations in btr0pcur.c

1. **Creating and Freeing a Persistent Cursor (`btr_pcur_create`, `btr_pcur_free`)**:

   - `btr_pcur_create()` allocates memory for a `btr_pcur_t` structure and initializes it.
   - `btr_pcur_free()` frees that memory and cleans up.
     This is straightforward memory management and initialization logic.

2. **Storing the Cursor Position (`btr_pcur_store_position`)**:
   When the cursor is positioned on a record, the function copies key information about that record (or a suitable prefix) into the cursor’s memory buffers. This stored information can later be used to re-locate this position after releasing and re-acquiring page latches.
   Notably:

   - If the index is empty, special flags indicate "before first" or "after last" positions.
   - If a valid record is under the cursor, the code stores the record prefix and maintain a “modify_clock” – a counter that can help detect if the page has changed since the position was stored.

3. **Copying Stored Position (`btr_pcur_copy_stored_position`)**:
   This function duplicates the stored position from one pcur to another. It copies all the stored record prefixes and states, enabling easy transfer of cursor positions.

4. **Restoring the Cursor Position (`btr_pcur_restore_position_func`)**:
   This is more complex. Once a persistent cursor’s position is stored, later the code tries to restore it. Restoration attempts:

   - First try an “optimistic restoration” if the page has not changed since the cursor position was stored (using the modify_clock and latch modes).
   - If that fails (the page changed or does not match expectations), it performs a new B-tree search to find the correct position again.

   The restoration ensures that even if the B-tree structure changed, the cursor will end up at a logically correct position corresponding to the old stored position, or as close as possible. If the exact record disappeared, it positions the cursor in a location consistent with the original request (e.g., next best position).

5. **Releasing Page Latches (`btr_pcur_release_leaf`)**:
   If the cursor currently holds a latch (S or X) on the leaf page, this function can release it. This step is crucial to ensure that the cursor does not block other threads. After releasing, the cursor remembers its position in memory but no longer has page access until restored later.

6. **Page-to-Page Moves (`btr_pcur_move_to_next_page` and `btr_pcur_move_backward_from_page`)**:
   These functions adjust the cursor when navigation needs to go beyond the current page (e.g., move to the next or previous page in the B-tree). They carefully release the old page latch and acquire a new latch on the target page, updating the stored position as needed. The code ensures that concurrency rules are not violated and that pages are properly latched/unlatched before and after moves.

7. **Open/Close and Navigate with a Persistent Cursor (`btr_pcur_open_on_user_rec_func`)**:
   This function is a wrapper to open a cursor on a user record that matches certain search conditions. Depending on the search mode (e.g., PAGE_CUR_GE, PAGE_CUR_LE), it finds the appropriate record. If no exact match is found, the cursor is positioned just before or after the requested key.

### Why This is Important

Persistent cursors increase concurrency and efficiency. Without persistent cursors, a database engine might need to keep pages latched for longer durations, blocking other transactions. Persistent cursors allow:

- **Scalability**: Releasing latches promptly is crucial when many transactions or queries run simultaneously.
- **Convenience**: By storing positions and restoring them later, code dealing with range scans, complex queries, or transactions that revisit data can be simplified.
- **Recovery from Changes**: If the tree structure changes (due to inserts, deletes, splits, merges), persistent cursors can still restore a logically close position without having to hold latches through the operation.

### Conclusion

`btr0pcur.c` and its persistent cursor logic are central to how early InnoDB versions managed and optimized B-tree navigation. The code snippet shows a system designed to hold and restore cursor positions in a complex, concurrent B-tree environment. Modern MySQL versions follow the same fundamental principles but with cleaner code, better documentation, and more robust concurrency features, making them easier to understand and maintain.

# btr0sea

## What is the Adaptive Hash Index?

InnoDB’s **Adaptive Hash Index (AHI)** is a hash-based in-memory structure built on frequently accessed B-tree pages. It works as follows:

- When InnoDB notices many repeated lookups of certain index ranges, it builds a hash index entry that allows subsequent lookups to jump directly to the target page, reducing I/O and CPU costs.
- The hash index is not persisted on disk; it is purely an in-memory optimization.
- InnoDB monitors access patterns to decide when and where to build these hash indexes. If usage patterns change, InnoDB may drop or rebuild them.

This code deals primarily with building, maintaining, and invalidating these hash indexes. It also contains logic for guessing a search position using hash indexes and verifying those guesses.

## Key Functional Areas in btr0sea.c

1. **Global Variables and Initialization**:

   - **`btr_search_enabled`**: A global flag indicating if the adaptive search system is currently enabled. The code provides functions to enable and disable it (`btr_search_enable()`, `btr_search_disable()`).
   - **`btr_search_latch`**: A special latch (RW-lock) protecting the AHI structures. All modifications to the hash index require holding this latch.
   - **`btr_search_sys`**: A global search system object that stores the main hash table (`hash_index`) and related metadata.
   - **`btr_search_sys_create()` and `btr_search_sys_close()`**: Functions to initialize and shut down the adaptive hash system at server start and stop.

2. **Building and Dropping Hash Indexes**:

   - **`btr_search_build_page_hash_index()`**: When certain criteria are met (e.g., a page is repeatedly accessed in a pattern suitable for a hash index), this function constructs a hash index for a page. It picks which fields/bytes of the key to use for the hash and inserts corresponding entries.
   - **`btr_search_drop_page_hash_index()`**: Removes an existing hash index for a particular page if no longer needed or if parameters have changed.

   The code ensures the correctness of the hash index by comparing fields and bytes of the index records and requires stable conditions (proper latches and checks).

3. **Search Position Guessing and Verification**:

   - **`btr_search_guess_on_hash()`**: This function tries to guess the position of a B-tree search using the hash index. If the guess is correct, the lookup avoids a full B-tree descent, improving performance.
   - **`btr_search_check_guess()`**: Verifies if the guessed position is correct by comparing the search tuple to records around the chosen record. If not correct, fallback to a normal B-tree search is required.

4. **Updating the Hash Index on Changes**:

   - **`btr_search_update_hash_on_insert()`, `btr_search_update_hash_on_delete()`**: When a record is inserted or deleted from a page, these functions adjust the hash index accordingly, ensuring that stale pointers are removed and new ones are added.
   - **`btr_search_move_or_delete_hash_entries()`**: Handles situations like page splits or merges, moving hash entries from one page to another or dropping them if pages are no longer relevant.

5. **Monitoring and Statistics**:

   - The code also includes counters and debugging facilities (`btr_search_validate()`), ensuring integrity and making it possible to detect corruption or inefficiencies in the hash structure.
   - Some performance counters (`btr_search_n_succ`, `btr_search_n_hash_fail`) help track how often the hash lookups succeed or fail, providing insights into the efficiency of the AHI.

6. **Concurrency and Latching Protocols**:

   - The adaptive hash index must be maintained with great care to prevent corruption. The code employs a dedicated latch `btr_search_latch` and strict latching order rules to avoid deadlocks and ensure consistency.
   - Buffer page latches are taken before or after the search latch as mandated by strict ordering rules. Many checks (`rw_lock_own()`) are sprinkled through the code as assertions to confirm correct latch usage.

## Why is This Important?

The adaptive hash index can greatly improve performance for certain types of workloads, especially when many queries access the same ranges of index keys repeatedly. By using a hash-based shortcut, InnoDB reduces the cost of locating data within the B-tree, speeding up lookups significantly.

However, building and maintaining these hash indexes costs CPU time and memory. Thus, this code includes logic to enable, disable, or rebuild hash indexes dynamically based on observed usage. Ensuring correctness, detecting stale entries, and validating assumptions is critical to maintain data consistency.

## Conclusion

`btr0sea.c` is a key part of the InnoDB adaptive search mechanism. It demonstrates how InnoDB builds an in-memory hash index on top of its B-tree pages, tries to guess search positions from hash lookups, and updates these indexes as pages and records change. The complexity of the code reflects the challenge of maintaining a high-performance, consistent, and concurrent in-memory structure that is tightly integrated with the InnoDB B-tree and buffer pool system.

While the fundamentals remain unchanged, newer MySQL versions present the same logic in a cleaner, more modular, and better-documented manner, making it easier for developers and users to understand, tune, and rely on the adaptive hash index for performance benefits.