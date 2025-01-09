# Files

Below is a high-level “road map” of how InnoDB source files (especially in older, embedded versions) are organized and how they relate to core InnoDB concepts. In InnoDB, the main subsystems include the buffer pool, B-tree index management, space/file management, transaction and lock management, row operations, logging/recovery, and various utility/OS layers. By grouping these files according to these subsystems (rather than describing them individually one by one), you’ll more easily see how each piece fits into InnoDB’s broader design.

---

## 1. B-Tree / Index Management
Files prefixed with `btr0` typically deal with InnoDB’s B+Tree indexes:

- **btr0btr.h / btr0btr.ic**: Core operations on B-tree pages (inserts, splits, page format).
- **btr0cur.h / btr0cur.ic**: Defines a “cursor” structure used to navigate the B-tree.
- **btr0pcur.h / btr0pcur.ic**: “Persistent cursor,” which remains valid across multiple operations.
- **btr0sea.h / btr0sea.ic**: B-tree searches and traversal logic.
- **btr0types.h**: Type definitions for B-tree structures (nodes, cursors, etc.).

**Why they matter:**  
These files represent the heart of how InnoDB organizes data on disk in a B+Tree, including how it finds and modifies rows by index key. If you want to see how rows are physically laid out on pages and navigated, the `btr0*` set of files is key.

---

## 2. Buffer Pool Management
These files manage InnoDB’s buffer pool and page caching:

- **buf0buddy.h / buf0buddy.ic**: The “buddy allocator” used internally for buffer blocks and memory segments.
- **buf0buf.h / buf0buf.ic**: Overall buffer pool structures and page caching logic.
- **buf0flu.h / buf0flu.ic**: Flushing logic (writing dirty pages from the buffer pool back to disk).
- **buf0lru.h / buf0lru.ic**: The LRU (least-recently-used) list management for buffer pages.
- **buf0rea.h**: Read-ahead logic in the buffer pool (prefetching pages).
- **buf0types.h**: Type definitions for the buffer pool subsystem.

**Why they matter:**  
Understanding how InnoDB caches pages in memory, manages dirty pages, and evicts pages (LRU) is central to grasping performance aspects and concurrency in the engine.

---

## 3. Space / File / Page Management
InnoDB organizes data in “tablespaces” and has its own space/file management layer:

- **fil0fil.h**: The file layer, referencing table space files and file segments.
- **fsp0fsp.h / fsp0fsp.ic**: InnoDB “fsp” stands for “file space.” Manages extent allocations, free lists, and general space management in a tablespace.
- **fsp0types.h**: Type definitions for file space management structures.
- **ibuf0ibuf.h / ibuf0ibuf.ic**: Insert buffer (change buffer) for secondary indexes.
- **ibuf0types.h**: Type definitions for insert buffer functionality.

**Why they matter:**  
All InnoDB data/metadata is stored in tablespaces that must handle free space, extents, and pages. Understanding the space management code helps you see how data is allocated on disk and how InnoDB organizes large data files.

---

## 4. Data / Row Operations
These files deal with row formats, how rows are inserted, updated, or read, and the row-related dictionary metadata:

- **data0data.h / data0data.ic**: Data field handling, row data abstraction (InnoDB row formats).
- **data0type.h / data0type.ic** and **data0types.h**: Definitions for different column/data types within InnoDB.
- **row0row.h / row0row.ic**: Basic row-level operations in the engine.
- **row0ins.h / row0ins.ic**: Row insertion logic.
- **row0upd.h / row0upd.ic**: Row update logic.
- **row0undo.h / row0undo.ic**: Undo logs and how row-level changes are rolled back.
- **row0sel.h / row0sel.ic**: Row selection (used in “SELECT” operations).
- **row0prebuilt.h**: “Prebuilt” row structures (query execution plan caching).
- **row0merge.h**: Merge operations for indexes (used by online DDL or maintenance).
- **row0types.h** / **rem0types.h**: Various row-related type definitions.

**Why they matter:**  
Crucial for understanding how rows are physically stored, how read/write operations happen at the row level, and how InnoDB handles changes to data.

---

## 5. Data Dictionary
These handle metadata about tables, indexes, columns, etc. InnoDB keeps a “dictionary” in its internal tables:

- **dict0dict.h / dict0dict.ic**: The central dictionary manager (creates/drops tables, indexes).
- **dict0boot.h / dict0boot.ic**: Bootstrapping the dictionary when InnoDB starts.
- **dict0crea.h / dict0crea.ic**: Dictionary creation logic for new tables/indexes.
- **dict0load.h / dict0load.ic**: Loading dictionary entries from system tables.
- **dict0mem.h / dict0mem.ic**: Memory structures for dictionary objects (tables, indexes).
- **dict0types.h**: Type definitions for dictionary objects.

**Why they matter:**  
The dictionary is how InnoDB knows about your tables and indexes (their columns, constraints, etc.). Every row operation, index operation, or transaction touches dictionary structures in some way.

---

## 6. Transaction & Locking System
These files manage concurrency via transactions and locks:

- **trx0trx.h / trx0trx.ic**: Core transaction handling structures and logic.
- **trx0undo.h / trx0undo.ic**: Undo logs for transactions; how versions of rows are tracked.
- **trx0roll.h / trx0roll.ic**: Rollback operations.
- **trx0purge.h / trx0purge.ic**: Purge operation that cleans up old row versions from undo logs once they’re no longer needed.
- **lock0lock.h / lock0lock.ic**: Record locks, table locks, and how locks are acquired or released.
- **lock0priv.h / lock0priv.ic** and **lock0types.h**: Lower-level locking and lock structure definitions.

**Why they matter:**  
InnoDB’s transaction model (ACID) depends on consistent reads, MVCC (multiversion concurrency control), and row-level locking. These files show exactly how InnoDB ensures transactional integrity, isolation, and concurrency.

---

## 7. Logging & Recovery
InnoDB keeps a redo log for crash recovery, plus logic for replaying changes:

- **log0log.h / log0log.ic**: Core redo logging structures (the log buffer, log file).
- **log0recv.h / log0recv.ic**: The recovery process (redo replay).
- **mach0data.h / mach0data.ic**: Low-level machine/byte-order operations used for logging and page operations.

**Why they matter:**  
If you want to understand how InnoDB recovers after a crash or ensures durability (the “D” in ACID), this is the place.

---

## 8. SQL Parser / Execution (Lightweight Embedded Pieces)
These are more specialized or “lightweight” parser/evaluator pieces historically used by the embedded version of InnoDB or for certain DDL operations:

- **pars0pars.h / pars0pars.ic**: Parser structures for SQL in the embedded engine.
- **pars0grm.h**: Grammar definitions.
- **pars0sym.h / pars0sym.ic**: Symbols and tokens.
- **pars0opt.h / pars0opt.ic**: Some optimization logic for the embedded parser.
- **pars0types.h**: Parser-related type definitions.
- **eval0eval.h / eval0eval.ic**: Simple expression evaluator.
- **eval0proc.h / eval0proc.ic**: Procedure-based evaluation logic.
- **que0que.h / que0que.ic**: Simple query execution structures (que = query).
- **que0types.h**: Type definitions for query structures.

**Why they matter:**  
In MySQL’s integrated version, most SQL parsing/execution is handled by the MySQL layer, but these embedded/standalone files let InnoDB do minimal SQL-like tasks internally (for metadata changes, replication logs, or older embedded uses).

---

## 9. OS/Portability Layer
These abstract away operating system–level operations so InnoDB can run on multiple platforms:

- **os0file.h**: File I/O abstractions (open/read/write).
- **os0thread.h / os0thread.ic**: Thread creation, synchronization, waiting.
- **os0sync.h / os0sync.ic**: Mutexes, semaphores, condition variables.
- **os0proc.h / os0proc.ic**: Process management wrappers (fork/exec, etc.).
- **os0types.h** (if present) would define OS-level types or wrappers.

**Why they matter:**  
InnoDB is cross-platform. This layer hides the raw OS calls so the higher-level engine code remains portable.

---

## 10. Miscellaneous / Utilities
A set of utility files that are heavily used throughout:

- **ut0byte.h / ut0byte.ic**: Byte-level utility functions (endianness, offset handling).
- **ut0rnd.h / ut0rnd.ic**: Random number utilities (e.g., for hashing or tests).
- **ut0list.h / ut0list.ic**: Linked list and list iterator utilities.
- **ut0mem.h / ut0mem.ic**: Low-level memory management wrappers.
- **ut0dbg.h**: Debugging macros.
- **ut0sort.h**: Sorting utilities.
- **ut0vec.h / ut0vec.ic**: Vector/array utilities.
- **fut0lst.h / fut0lst.ic**: “fut” stands for “free list” or “future list,” used for maintaining lists of free objects.

**Why they matter:**  
Much of the “glue” that holds the engine together is in these utility functions. They’re invoked everywhere for low-level data structures like lists, memory debugging, random data, etc.

---

## 11. Other Notable Files
- **api0api.h, api0misc.h, api0ucode.h**: “API” stubs for the embedded engine or older special function calls.  
- **ha0ha.h / ha0ha.ic, ha0storage.h / ha0storage.ic**: “ha” means “handler”—integration points for the MySQL layer (or a custom SQL layer) to call into InnoDB.  
- **mtr0mtr.h / mtr0mtr.ic**: “mini-transaction” logic—InnoDB’s internal short-duration atomic operations (especially for page modifications).  
- **thr0loc.h / thr0loc.ic**: Thread-local storage or thread-based logic.  
- **sync0arr.h / sync0arr.ic, sync0rw.h / sync0rw.ic, sync0sync.h / sync0sync.ic, sync0types.h**: InnoDB synchronization arrays, read-write locks, sync primitives.  
- **ib0config.h**: Build or configuration settings for the embedded InnoDB.

---

## How to Use This Road Map
1. **Start with the fundamentals**: If you’re new to InnoDB internals, begin with how the **transaction system** (`trx0*`) and **row operations** (`row0*`) work.  
2. **Move to B-tree and buffer pool**: Learn how data is stored on pages (`btr0*`) and how those pages are cached in memory (`buf0*`).  
3. **Add logging/recovery**: Understand how modifications are persisted (`log0*`) and undone/purged (`trx0undo`, `trx0purge`).  
4. **Tie in dictionary**: See how InnoDB keeps table and index metadata consistent (`dict0*`).  
5. **Finally, OS and utility layers**: Dive into the details of synchronization (`os0sync`, `sync0*`), file I/O (`os0file`), plus all the small utility files (`ut0*`).

By grouping the files into these subsystems, you’ll have a clearer picture of how each piece of the engine works, how they interact, and what order makes sense to learn them in. Understanding these groupings (rather than line-by-line file descriptions) is the key to mastering InnoDB internals.

# Structures

Below is a summary of some of **the most critical data structures** you’ll encounter in InnoDB. Each structure is fundamental to understanding how the engine manages data, transactions, and recovery. Where possible, I’ve included references to the files where these structures are primarily declared or used.

---

## 1. Transaction and Locking Structures

1. **trx_t**  
   - **Where:** Declared in *trx0trx.h*  
   - **What it is:** The core transaction descriptor. Tracks a transaction’s state, isolation level, undo segments, etc.  
   - **Why it matters:** Almost every operation in InnoDB is performed under a transaction context, so this structure is central to concurrency, MVCC, and ACID compliance.

2. **trx_undo_t**  
   - **Where:** Declared in *trx0undo.h*  
   - **What it is:** Manages the undo logs for a given transaction. Contains references to old row versions, undo segments, etc.  
   - **Why it matters:** Undo logs store “before images” of updated rows, enabling rollback and consistent reads.

3. **trx_rseg_t** (Rollback Segment)  
   - **Where:** *trx0rseg.h*  
   - **What it is:** Represents a rollback segment, a space in the datafile containing multiple undo logs.  
   - **Why it matters:** InnoDB uses multiple rollback segments to handle concurrent transactions.

4. **lock_t**  
   - **Where:** *lock0lock.h*  
   - **What it is:** Represents a single lock—either row-level or table-level.  
   - **Why it matters:** Lock structures handle row locking (shared/exclusive) and table locks for DDL statements.

5. **lock_sys_t**  
   - **Where:** *lock0lock.h*  
   - **What it is:** The global lock system descriptor, containing hash tables for locked rows, arrays for lock objects, etc.  
   - **Why it matters:** Central place for InnoDB to track active locks and check conflicts.

---

## 2. B-Tree / Index Structures

1. **btr_cur_t** (B-Tree Cursor)  
   - **Where:** *btr0cur.h* / *btr0cur.ic*  
   - **What it is:** A cursor that navigates an InnoDB B+Tree (used to search, insert, or delete rows).  
   - **Why it matters:** Allows row-level operations (search, next record, etc.) on pages.

2. **btr_pcur_t** (Persistent B-Tree Cursor)  
   - **Where:** *btr0pcur.h* / *btr0pcur.ic*  
   - **What it is:** Similar to btr_cur_t but remains valid across multiple operations.  
   - **Why it matters:** Useful for operations that span multiple steps (e.g., scanning an index).

3. **page_t** and **page_header_t**  
   - **Where:** Typically found in *page0page.h* / *page0page.ic*  
   - **What it is:** Low-level structures that define the format/layout of an InnoDB page (size, page header, directory slots, etc.).  
   - **Why it matters:** Every row and index in InnoDB is stored on “pages” (usually 16KB). Understanding page structure is key to B-tree logic.

---

## 3. Buffer Pool Structures

1. **buf_pool_t**  
   - **Where:** *buf0buf.h*  
   - **What it is:** The global buffer pool descriptor, containing all buffer control blocks, LRU lists, etc.  
   - **Why it matters:** Central to how InnoDB caches disk pages in memory, tracks dirty pages, and implements the LRU mechanism.

2. **buf_block_t** (sometimes also referred to as buf_page_t)  
   - **Where:** *buf0buf.h*  
   - **What it is:** Represents an individual page in the buffer pool, including the page frame, fix count, and a pointer to the actual data.  
   - **Why it matters:** The fundamental “unit” of caching and page access in the buffer pool.

3. **LRU-related lists** (e.g., `buf_page_t::list` or the LRU list structures)  
   - **Where:** *buf0lru.h* / *buf0lru.ic*  
   - **What they are:** Data structures for maintaining pages in LRU order, used for eviction.  
   - **Why they matter:** Determine how pages are aged out of memory, a key performance factor.

---

## 4. Dictionary / Metadata Structures

1. **dict_table_t**  
   - **Where:** *dict0dict.h*  
   - **What it is:** Represents a table in the internal dictionary (metadata about columns, indexes, constraints).  
   - **Why it matters:** Everything from row format to indexes is managed via `dict_table_t`.

2. **dict_index_t**  
   - **Where:** *dict0dict.h*  
   - **What it is:** Describes an index belonging to a table (B+Tree root page, unique flags, fields, etc.).  
   - **Why it matters:** InnoDB uses one “clustered index” (primary) plus optional secondary indexes. This structure records index-level details.

3. **dict_col_t**  
   - **Where:** *dict0dict.h* or *dict0mem.h*  
   - **What it is:** Metadata for a single column (type, length, name).  
   - **Why it matters:** InnoDB needs column definitions for row parsing, constraints, etc.

---

## 5. Row / Record-Level Structures

1. **row_prebuilt_t**  
   - **Where:** *row0prebuilt.h*  
   - **What it is:** A pre-built “execution” structure containing pointers for row operations (used by MySQL or the embedded query interface).  
   - **Why it matters:** Optimizes repeated operations by caching pointers to indexes and row format details.

2. **row_ext_t** / **row_merge_buf_t** / **row_merge_block_t**  
   - **Where:** *row0ext.h*, *row0merge.h*  
   - **What they are:** Structures related to external storage of columns (ROW_FORMAT=DYNAMIC/COMPRESSED) or merges for online DDL.  
   - **Why they matter:** Show how InnoDB handles big columns (BLOB/TEXT) stored off-page or reorganizes data for index creation.

3. **rem0rec.h** (Record format)  
   - **Where:** *rem0rec.h* / *rem0rec.ic*  
   - **What it is:** Low-level record (row) format on a page, including offsets and variable-length fields.  
   - **Why it matters:** InnoDB’s on-page record format is the basis for reading/writing physical rows in the B-tree.

---

## 6. Mini-Transaction (MTR) Structures

1. **mtr_t** (Mini-Transaction)  
   - **Where:** *mtr0mtr.h* / *mtr0mtr.ic*  
   - **What it is:** A short-lived, atomic operation context that makes changes to pages in memory. These changes become visible once the MTR commits.  
   - **Why it matters:** InnoDB uses MTRs to ensure internal consistency for single-page or small multi-page changes that must be atomic from the engine’s perspective.

2. **mtr_buf_t** / **mtr_memo_t**  
   - **Where:** *mtr0mtr.h*  
   - **What they are:** Buffers and memory structures that the mini-transaction uses to log changes to pages (i.e., for redo logging).  
   - **Why they matter:** Provide the “before” and “after” states for operations, which are later written to the redo log.

---

## 7. Logging and Recovery Structures

1. **log_t**  
   - **Where:** *log0log.h*  
   - **What it is:** Global log descriptor, containing the log buffer, log files, and related info.  
   - **Why it matters:** InnoDB’s redo log ensures changes are durable and recoverable after a crash.

2. **log_rec_t**  
   - **Where:** *log0recv.h* or *log0log.h*  
   - **What it is:** A single log record describing a page change.  
   - **Why it matters:** The fundamental unit of redo logging and crash recovery.

3. **log_sys_t**  
   - **Where:** *log0log.h*  
   - **What it is:** The global logging system, containing pointers to log files, the log buffer, the next log sequence number, etc.  
   - **Why it matters:** Orchestrates how new records get appended to the log and how the log is flushed to disk.

---

## 8. Space / File Management Structures

1. **fil_space_t**  
   - **Where:** *fil0fil.h*  
   - **What it is:** Describes a single tablespace (file name, space ID, size info, etc.).  
   - **Why it matters:** InnoDB can have multiple tablespaces; each is tracked by a `fil_space_t`.

2. **fsp_header_t** (File Space Header)  
   - **Where:** *fsp0fsp.h*  
   - **What it is:** The on-disk header page structure that tracks free extents, segment info, etc.  
   - **Why it matters:** InnoDB must manage data file extents, allocations, and free space at the page level.

3. **ibuf_t** (Insert/Change Buffer)  
   - **Where:** *ibuf0ibuf.h*  
   - **What it is:** Tracks buffered changes to secondary indexes that are applied lazily.  
   - **Why it matters:** Improves performance by deferring random I/O for secondary index updates (especially for non-unique indexes).

---

## 9. Utility and OS-Abstraction Structures

1. **ut_list_node_t** / **UT_LIST_BASE_NODE_T**  
   - **Where:** *ut0list.h* / *ut0list.ic*  
   - **What they are:** Doubly-linked list node and base macros used throughout InnoDB for linked lists.  
   - **Why they matter:** InnoDB uses UT_LIST for buffer lists, lock lists, etc.

2. **os_event_t**, **os_mutex_t**, **os_cond_t**, etc.  
   - **Where:** *os0sync.h* / *os0sync.ic*  
   - **What they are:** Wrappers for OS-level synchronization constructs (mutexes, events, condition variables).  
   - **Why they matter:** Provide a portable layer for multi-threaded concurrency on different operating systems.

3. **mtr_memo_slot_t**  
   - **Where:** *mtr0mtr.h*  
   - **What it is:** Part of the mini-transaction system’s memory management for tracking pages and blocks changed under the MTR.  
   - **Why it matters:** Ensures changes can be rolled back or redone if the MTR fails or commits.

---

## Putting It All Together
- **`trx_t`** and **`btr_cur_t`** are likely the single best “jumping-off” points** to see how InnoDB executes an operation within a transaction context while navigating or modifying a B-tree.  
- **`buf_block_t`** and **`page_t`** show how pages are structured in memory and on disk, respectively.  
- **`dict_table_t`** and **`dict_index_t`** explain how InnoDB tracks metadata for each table and index.  
- **`log_rec_t`** and **`trx_undo_t`** illustrate how changes are recorded in redo/undo logs for crash recovery and MVCC.  
- **`mtr_t`** (mini-transaction) ties everything together at the page-operation level, ensuring atomic changes.

Familiarizing yourself with these key structures and how they’re declared/used in the corresponding header (`*.h`) and inline code (`*.ic`) files will go a long way toward helping you understand the overall architecture of InnoDB.
