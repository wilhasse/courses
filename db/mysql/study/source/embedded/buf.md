# Concepts

Below is a line-by-line explanation (with additional commentary and simplified examples) of what the InnoDB source code comment in **buf0buf.c** is talking about. The commentary tries to “translate” the developer notes into more digestible terms, and give you a big-picture understanding of how the buffer pool, control blocks, lists, and read-ahead mechanisms work within InnoDB.

---

## Performance Improvement Notes

```c
Thread scheduling in NT may be so slow that the OS wait mechanism should
not be used even in waiting for disk reads to complete. Rather, we should
put waiting query threads to the queue of waiting jobs, and let the OS
thread do something useful while the i/o is processed.
```

### Explanation

- These comments date back to when InnoDB was first ported to Windows NT (an older Windows operating system). 
- The authors note that thread scheduling on Windows could be slow. Instead of having a query thread simply “sleep” or block while waiting for I/O to finish, it might be more efficient to have that thread do other useful work until the disk I/O is ready.
- This is basically referencing **cooperative multitasking** or using “async” style I/O where you don’t block a thread while a disk operation is pending.

### Example

- **Traditional**: Thread A requests a page read from disk. Thread A blocks, waiting. The operating system does a context switch to some other thread, etc.
- **Proposed**: Thread A requests a page read from disk and places itself in a queue of “waiting for I/O.” Meanwhile, Thread A can do other tasks or the scheduler can easily move on without a heavy context switch penalty. Once the disk read is completed, Thread A is “unblocked” with minimal overhead.

---

```c
A possibility is to put a user space thread library between the database
and NT. User space thread libraries might be very fast.
```

### Explanation

- The developer is suggesting that a user-space thread library (like fibers or “green threads”) could be used. User-space threads can switch faster than kernel-level threads because they avoid entering the kernel scheduler.

---

```c
SQL Server 7.0 can be configured to use 'fibers' which are lightweight
threads in NT. These should be studied.
```

### Explanation

- This is just an acknowledgement that SQL Server was already doing something like this using “fibers.”  
- Fibers are a form of **user-mode scheduling** on Windows.

---

## Buffer Frames and Blocks

```c
Following the terminology of Gray and Reuter, we call the memory blocks
where file pages are loaded buffer frames. For each buffer frame there
is a control block, or shortly, a block, in the buffer control array.
The control info ... resides in the control block.
```

### Explanation

- In InnoDB, **“buffer frames”** are the actual areas in memory that hold data pages (e.g., 16 KB pages if you’re using a default InnoDB page size).
- For every “buffer frame” (i.e., that 16 KB chunk of memory), there is a corresponding **control block** in an array. The control block is essentially a small C struct that keeps track of metadata: 
  - Which page is in this frame?
  - Whether the page is dirty (modified) or not?
  - Reference counters, read/write locks, etc.

### Example

- Suppose you have 100 buffer frames (100 × 16 KB). You will also have 100 control blocks in the “buffer control array.” Each control block tracks usage details about its corresponding 16 KB frame.

---

## The Buffer Pool Struct

```c
The buffer buf_pool contains a single mutex which protects all the
control data structures of the buf_pool. ...
```

### Explanation

- **`buf_pool`** is the main data structure for InnoDB’s buffer pool. It has a single, large mutex (`buf_pool->mutex`) that guards critical sections and internal data structures (like the LRU list, hash tables, free lists, etc.).
- Additionally, each buffer frame’s contents are protected by its own read-write lock. 
  - So even though there is one “big” mutex for orchestrating global changes, the actual data in a page can be protected separately by a lighter weight lock.

### Why is this important?

- Having a single “big” mutex can cause contention on multi-CPU systems. So the developers were aware that you want to minimize the time spent holding that global buffer pool mutex.

---

```c
The buf_pool mutex is a hot-spot in main memory, causing a lot of
memory bus traffic on multiprocessor systems ...
```

### Explanation

- On older multiprocessor architectures (e.g., multi-socket Pentiums in the 1990s and 2000s), constantly taking and releasing a single global lock causes a lot of bus traffic and slows down performance when many threads are all contending for that lock.

---

```c
A solution to reduce mutex contention of the buf_pool mutex is to
create a separate mutex for the page hash table ...
```

### Explanation

- One approach to reduce contention is to break out certain data structures (like the “page hash table” that maps `(space_id, page_number)` → control block) into separate locks, so you don’t always have to lock everything under the big `buf_pool->mutex`.

---

## Control Blocks

```c
The control block contains, for instance, the bufferfix count ...
For this purpose, the control block contains a read-write lock.
```

### Explanation

- **Bufferfix count**: an integer that says “how many threads are currently pinning (fixing) this page in memory?” 
- The **read-write lock** in the control block lets multiple readers “lock” the page for reading or a single thread “lock” it for writing. This is separate from the big `buf_pool->mutex`.

---

```c
We intend to make the buffer buf_pool size on-line reconfigurable ...
If the buf_pool size is cut, we exploit the virtual memory mechanism of
the OS ...
```

### Explanation

- The developers wanted to allow resizing the buffer pool at runtime. A smaller buffer pool means less memory in use; a bigger buffer pool means more memory. Rather than physically removing allocated blocks, InnoDB could simply stop using them so the OS can page them out as needed.

---

```c
The control blocks containing file pages are put to a hash table
according to the file address of the page. ...
```

### Explanation

- Inside the buffer pool, there is a **hash table** that maps a “page identifier” (space/file + page number) to the control block in memory that contains that page. That way, when you ask for a certain page, InnoDB can quickly find if it’s in memory or not.
- The developers toyed with the idea of “pointer swizzling” (replacing page references in B-tree nodes with direct memory pointers), but decided it’s simpler to keep an efficient hash table.

---

## Lists of Blocks

```c
The free list (buf_pool->free) contains blocks which are currently not
used.
```

### Explanation

- If a control block (and its associated buffer frame) is not storing any data page at the moment, it lives in the **free list**.  
- When you need a new page, you first try to grab from the **free list**. If that’s empty, you might have to evict pages from the LRU list (described next).

---

```c
The common LRU list contains all the blocks holding a file page except
those for which the bufferfix count is non-zero.
```

### Explanation

- **LRU = Least Recently Used.**  
- All pages that are actively in use in the buffer pool (and not currently pinned or “fixed” by a thread) go into this LRU list.  
- The LRU list is kept in order of how recently each page was accessed. The “least recently accessed” block sits toward the tail.

### Example

- You have 10 pages in the buffer pool. Whenever a page is accessed, it moves to the “front” (or “hot end”) of the LRU list. Pages that haven’t been accessed in a while drift to the “back.” When you need to make space, you remove pages from the back of the LRU list.

---

```c
The unzip_LRU list contains a subset of the common LRU list ...
```

### Explanation

- This relates to **compressed InnoDB pages**.  
- If a page is compressed, InnoDB might keep both the compressed version and an uncompressed version. The `unzip_LRU` is a special list that tracks pages which currently exist in uncompressed form.
- The order in `unzip_LRU` mirrors the order in the main LRU. Whenever an entry moves in the main LRU, it also moves in the `unzip_LRU` if that page is compressed.

---

```c
The chain of modified blocks (buf_pool->flush_list) contains the blocks
holding file pages that have been modified in the memory but not written
to disk yet. ...
```

### Explanation

- **Flush List**: All “dirty” pages that must eventually be written back to disk.  
- The flush list is ordered roughly by the time of the oldest modification. This helps InnoDB decide which pages to flush to disk first (older changes get flushed out in a checkpointing process).

---

```c
The chain of unmodified compressed blocks (buf_pool->zip_clean) ...
```

### Explanation

- If a compressed page in memory is **not** dirty, it can be on this `zip_clean` list, meaning it holds only the compressed version and no uncompressed buffer.

---

## Loading a File Page

```c
First, a victim block for replacement has to be found ... from the free list
or from the end of the LRU-list. An exclusive lock is reserved for the frame,
the io_fix field is set ...
```

### Explanation

- When InnoDB needs to load a page from disk:
  1. It looks for a **victim** (a free buffer or a buffer at the tail of the LRU to evict).  
  2. It **locks** that block exclusively and marks `io_fix` to indicate “this block is in the middle of an I/O operation.”  
  3. The disk read is issued. Once the I/O finishes, the block’s exclusive lock is released, `io_fix` is reset, and the page is considered loaded.

### Example

- You query a row that resides in page X, but page X is not in memory. InnoDB picks a block from the free list. If the free list is empty, it evicts the oldest page from the LRU. That block is locked, read from disk into the block, then unlocked. Now page X is in memory.

---

## Read-Ahead Mechanism

```c
The read-ahead mechanism is intended to be intelligent and isolated from
the semantically higher levels of the database ...
```

### Explanation

- **Read-ahead** is about predicting which pages you might need soon. If InnoDB notices a certain access pattern, it pre-emptively schedules I/O for pages near the pages you’re loading so that future reads are faster (since the OS can batch disk I/O or fetch them sequentially).

### Two main read-ahead approaches:

1. **Linear read-ahead**  
   - For example, if the system notices you’re scanning pages in ascending or descending order of page IDs (like scanning the leaf level of a B-tree).  
   - The tablespace is divided into areas (say, blocks 0–63, 64–127, etc.). If it detects you just read all the pages in one area, it predicts you’ll read the next area. Then it issues a read-ahead for that next chunk of pages.

2. **Random read-ahead**  
   - If the access pattern is not strictly sequential but the server notices multiple pages within a certain consecutive range (say 32 pages) were recently accessed, it might guess that you’ll need the rest of that range soon. It schedules a bulk read of that range.

### Why?

- Reading from disk is much slower than memory, but if you do large, **sequential** reads, you can often get big throughput gains (especially on spinning disks). Even with SSDs, read-ahead can help if it reduces the overhead of multiple discrete I/O operations.

---

# Putting It All Together

- **Buffer pool** is at the core of InnoDB’s caching layer. 
- **Control blocks** are the metadata for each in-memory page. 
- A **global buffer pool mutex** serializes many operations, but each page’s content has its own read-write lock. 
- Pages are organized in multiple **lists**:
  - **Free list** (unused)
  - **LRU list** (in-memory pages, ordered by recency of access)
  - **Flush list** (dirty pages awaiting disk writes)
  - **zip_clean** list (compressed & clean pages)
  - **unzip_LRU** (tracks uncompressed versions of compressed pages)
- **Read-ahead** attempts to fetch pages from disk in bulk once InnoDB anticipates a certain sequential or random pattern.

## Example Scenario

1. **You run a large table scan** (e.g., `SELECT * FROM table` without an index):
   - InnoDB sees that you are reading pages 1, 2, 3, 4, … in ascending order.  
   - After it detects you’ve read all pages in the 0–63 range, it issues a read-ahead of pages 64–127 in the background.

2. **Buffer Pool Eviction**:
   - The LRU list might start to fill. Pages 20, 21, 22 have been used recently, so they move up in the LRU list.  
   - Pages that haven’t been touched in a long time drift to the tail.  
   - If you need to read a new page from disk, InnoDB might evict the page at the tail of the LRU list (say page 5) to reuse that space.

3. **Dirty Page Flush**:
   - Meanwhile, a transaction updated page 1000. Page 1000 is placed on the flush list.  
   - The background flusher sees page 1000 is older than some threshold, writes it to disk, and removes it from flush list (or marks it as clean).

4. **Read-Write Lock**:
   - If two sessions read from the same page, they do a shared (read) lock on the control block.  
   - If something needs to modify the page, it requests an exclusive (write) lock on that page’s control block.

---

### Key Takeaways

1. **Minimize Contention**: The developers mention how frequent global mutex acquisitions can hurt performance, especially in multi-CPU environments. They considered more granular locks, user-space threads, or other optimizations.
2. **Data Structures**: InnoDB organizes its buffer pool via several lists (LRU, free, flush, etc.) and a hash table for quick lookups.
3. **On-the-fly Page Management**: Once a page is requested, it must be located or loaded from disk, pinned, and eventually unpinned. Dirty pages go on the flush list and are written back later.
4. **Read-Ahead**: InnoDB tries to be “smart” about upcoming pages so that sequential or semi-random access patterns can be optimized by prefetching.

# buf0buf.c

Below is an **annotated walkthrough** of the major parts of the code you shared from **buf0buf.c**. This file implements a significant portion of InnoDB’s **buffer pool** logic, including:

- Creating and destroying the buffer pool.
- Allocating and managing pages and chunks of memory (the “frames”).
- Calculating and verifying page checksums.
- Reading and writing pages from/to disk (I/O handling).
- Handling compressed pages when **WITH_ZIP** is enabled.
- Dealing with LRU, free, and flush lists.
- Debugging, validation, and statistics gathering.

I’ve broken it down into sections that mirror the code’s structure. Each section references specific functions and data structures with a high-level explanation of what they do and why they matter.

---

## 1. Includes and Global Variables

```c
#include "buf0buf.h"
...
/** Value in microseconds */
UNIV_STATIC const int WAIT_FOR_READ = 5000;
/** Number of attempts made to read in a page in the buffer pool */
static const ulint BUF_PAGE_READ_MAX_RETRIES = 100;

/** The buffer buf_pool of the database */
UNIV_INTERN buf_pool_t* buf_pool = NULL;

/** mutex protecting the buffer pool struct and control blocks */
UNIV_INTERN mutex_t buf_pool_mutex;
#ifdef WITH_ZIP
UNIV_INTERN mutex_t buf_pool_zip_mutex;
#endif /* WITH_ZIP */
...
```

### Explanation

1. **Includes**: The file pulls in headers for memory (`mem0mem.h`), B-tree operations (`btr0btr.h`), file I/O (`fil0fil.h`), concurrency structures (`lock0lock.h`), logging (`log0log.h`), and so on.  
2. **Global variables**:
   - `buf_pool`: A pointer to the global InnoDB buffer pool (type `buf_pool_t`).
   - `buf_pool_mutex`: A global mutex that protects most of the shared data structures in the buffer pool.
   - `buf_pool_zip_mutex`: An additional mutex used specifically for compressed pages (when `WITH_ZIP` is enabled).
   - Constants like `WAIT_FOR_READ` (used for short sleeps while waiting for asynchronous I/O) and `BUF_PAGE_READ_MAX_RETRIES` (if a page repeatedly fails to load, it stops trying after this many attempts).

---

## 2. Basic Initialization and Checksums

### **`buf_var_init()`**
```c
UNIV_INTERN
void
buf_var_init(void)
{
    buf_pool = NULL;
    memset(&buf_pool_mutex, 0x0, sizeof(buf_pool_mutex));
#ifdef WITH_ZIP
    memset(&buf_pool_zip_mutex, 0x0, sizeof(buf_pool_zip_mutex));
#endif
    ...
}
```
**Purpose**: Resets global buffer-related variables to a safe initial state (used early in the server startup or tests).

---

### **`buf_calc_page_new_checksum()`** and **`buf_calc_page_old_checksum()`**
```c
UNIV_INTERN
ulint
buf_calc_page_new_checksum(const byte* page) { ... }

UNIV_INTERN
ulint
buf_calc_page_old_checksum(const byte* page) { ... }
```
**Purpose**: InnoDB can store **two** types of checksums on each page:

1. **New formula**: Spread out across the page fields, skipping certain bytes (like the log sequence number, or LSN).
2. **Old formula**: Historical reasons; older MySQL/InnoDB versions used this simpler method.

These functions carefully compute each type of checksum, ensuring 32-bit consistency across platforms (32-bit or 64-bit).

---

### **`buf_page_is_corrupted()`**
```c
UNIV_INTERN
ibool
buf_page_is_corrupted(const byte* read_buf, ulint zip_size) {
    ...
    /* Compares page checksums, LSN fields, etc. */
    ...
    return(FALSE);
}
```
**Purpose**: Checks if a page read from disk has mismatched checksums or invalid LSN values, indicating corruption or a failed read.  
- If the page is invalid, it returns `TRUE`.  
- Also checks that certain fields (like the LSN at the beginning and end of the page) match.

---

## 3. Debug Printing of Pages

### **`buf_page_print()`**
```c
UNIV_INTERN
void
buf_page_print(const byte* read_buf, ulint zip_size) {
    /* Prints page data in hex and ASCII, plus
       relevant metadata for debugging. */
    ...
}
```
**Purpose**: If InnoDB detects corruption or other debugging scenarios, it can print out the contents of the page, including checksums, LSNs, space and page IDs, etc. This is extremely useful in diagnosing page failures.

---

## 4. Buffer Pool Data Structures and Initialization

### **`buf_pool_t`** and **`buf_chunk_t`**
In the code, we see:

```c
struct buf_chunk_struct {
    ulint mem_size;
    ulint size;
    void* mem;
    buf_block_t* blocks;
};
...
UNIV_INTERN buf_pool_t* buf_pool = NULL;
```

- A **`buf_pool_t`** object manages:
  - Arrays (or lists) of **buffer frames** (the actual page-sized allocations).
  - Global counters, events, and lists (LRU, free list, flush list, etc.).
  - Mutexes and other concurrency structures.

- A **`buf_chunk_t`** represents a contiguous “chunk” of memory allocated for storing multiple pages. Each chunk has:
  - `mem_size`: how many bytes were actually allocated.
  - `size`: how many page frames fit in that chunk.
  - `mem`: the start address of that chunk.
  - `blocks`: an array of `buf_block_t` control blocks, each describing a single page frame.

---

### **`buf_block_init()`** and **`buf_chunk_init()`**
```c
UNIV_STATIC
void
buf_block_init(buf_block_t* block, byte* frame) {
    ...
    block->frame = frame;
    block->page.state = BUF_BLOCK_NOT_USED;
    block->page.buf_fix_count = 0;
    ...
    /* Create a per-block mutex and rw_lock */
    mutex_create(&block->mutex, SYNC_BUF_BLOCK);
    rw_lock_create(&block->lock, SYNC_LEVEL_VARYING);
    ...
}

UNIV_STATIC
buf_chunk_t*
buf_chunk_init(buf_chunk_t* chunk, ulint mem_size) {
    ...
    chunk->mem = os_mem_alloc_large(&chunk->mem_size);
    ...
    chunk->blocks = chunk->mem;
    ...
    /* Initialize each buf_block_t in this chunk. */
    for (i = chunk->size; i--; ) {
        buf_block_init(block, frame);
        ...
        /* Add the block to the free list. */
        UT_LIST_ADD_LAST(list, buf_pool->free, (&block->page));
        ...
        block++;
        frame += UNIV_PAGE_SIZE;
    }
    return(chunk);
}
```
**Purpose**:
1. `buf_block_init()`: Sets up a single `buf_block_t` control structure, linking it to its frame (16 KB of memory by default). Also creates a mutex and read-write lock used when accessing the page’s data.
2. `buf_chunk_init()`: Allocates a big memory area (`mem_size`), divides it into pages, and initializes a `buf_block_t` for each page. Each block is then placed on the **free list** of the buffer pool.

---

### **`buf_pool_init()`**
```c
UNIV_INTERN
buf_pool_t*
buf_pool_init(void) {
    buf_pool = mem_zalloc(sizeof(buf_pool_t));
    mutex_create(&buf_pool_mutex, SYNC_BUF_POOL);
#ifdef WITH_ZIP
    mutex_create(&buf_pool_zip_mutex, SYNC_BUF_BLOCK);
#endif
    ...
    buf_pool_mutex_enter();
    buf_pool->n_chunks = 1;
    buf_pool->chunks = chunk = mem_alloc(sizeof *chunk);
    UT_LIST_INIT(buf_pool->free);
    if (!buf_chunk_init(chunk, srv_buf_pool_size)) {
        ...
        return(NULL);
    }
    ...
    buf_pool->curr_size = chunk->size;
    ...
    buf_pool->page_hash = hash_create(2 * buf_pool->curr_size);
#ifdef WITH_ZIP
    buf_pool->zip_hash = hash_create(2 * buf_pool->curr_size);
#endif
    ...
    buf_pool_mutex_exit();
    ...
    return(buf_pool);
}
```
**Purpose**:
- Allocates the global `buf_pool_t`.
- Creates the buffer pool mutexes.
- Allocates a single “chunk” for now (though InnoDB can be configured to have multiple chunks) of size `srv_buf_pool_size`.
- Sets up hash tables (`page_hash`, and optionally `zip_hash`) for looking up pages by `(space_id, page_no)`.
- Prepares free lists and LRU lists.

When complete, we have a functioning buffer pool in memory with pages ready for use.

---

## 5. Page Access and I/O: `buf_page_get_gen()`, `buf_page_init_for_read()`, etc.

### **`buf_page_get_gen()`**
```c
UNIV_INTERN
buf_block_t*
buf_page_get_gen(
    ulint space, ulint zip_size, ulint offset, ulint rw_latch,
    buf_block_t* guess, ulint mode, const char* file, ulint line, mtr_t* mtr)
{
    ...
    for (;;) {
        buf_pool_mutex_enter();
        ...
        block = (buf_block_t*) buf_page_hash_get(space, offset);
        if (!block) {
            /* Page not in buffer -> schedule read */
            buf_pool_mutex_exit();
            if (mode != BUF_GET_IF_IN_POOL) {
                /* read from disk */
                buf_read_page(space, zip_size, offset);
            }
            ...
            continue; /* Retry after reading. */
        }
        ...
        /* If it's currently I/O-fixed by another read, wait or exit. */
        ...
        /* "Latch" the page with S or X latch, etc. */
        ...
        /* Update LRU status (make the page "young" if recently used). */
        ...
        return(block);
    }
}
```
**Purpose**:
1. Look up the requested `(space, page_no)` in the buffer pool’s page hash.
2. If it’s **not** there, the function optionally calls `buf_read_page()` to schedule an asynchronous read from disk.
3. If it **is** there but is still in the middle of being read (`io_fix == BUF_IO_READ`), it waits or returns `NULL` if `BUF_GET_IF_IN_POOL` mode is used.
4. Once the page is in memory and not pinned by I/O, the caller obtains a read (`RW_S_LATCH`) or write (`RW_X_LATCH`) latch on it (unless `mode == BUF_GET_NO_LATCH`, i.e., no latch requested).
5. Adjusts LRU ordering so that recently accessed pages move toward the “new” or “young” end.

**Key Points**:
- The function returns a **buffer-fixed** page (i.e., `page->buf_fix_count++`).
- Once done, the caller uses `mtr_commit()` or buffer fix decrement operations, eventually unlocking the page so others can use it.

---

### **`buf_page_init_for_read()`**
```c
UNIV_INTERN
buf_page_t*
buf_page_init_for_read(
    ulint* err, ulint mode, ulint space, ulint zip_size,
    ibool unzip, ib_int64_t tablespace_version, ulint offset)
{
    ...
    if (buf_page_hash_get(space, offset)) {
        /* The page is already in the buffer. Nothing to do. */
        ...
        return(NULL);
    }
    /* Otherwise, pick a free block or a victim from LRU. */
    block = buf_LRU_get_free_block(0);
    ...
    /* Insert it into page hash, LRU list, set io_fix=BUF_IO_READ, and
       X-latch the block so no one else can read it until the I/O is done. */
    ...
    buf_pool->n_pend_reads++;
    ...
    return(&block->page);
}
```
**Purpose**:
- This is a lower-level function used by the asynchronous I/O subsystem to “reserve” a block for reading a specific page into memory.
- Sets `io_fix = BUF_IO_READ` and an exclusive latch so no one else can treat the block as valid until the data is loaded from disk.

---

### **`buf_page_io_complete()`**
```c
UNIV_INTERN
void
buf_page_io_complete(buf_page_t* bpage) {
    /* Called after the I/O thread finishes reading or writing a page. */
    ...
    if (io_type == BUF_IO_READ) {
        /* If read is complete, check checksums, possibly do
           a crash recovery hook (recv_recovery_is_on()). */
        ...
        if (!recv_no_ibuf_operations) {
            ibuf_merge_or_delete_for_page(...);
        }
        buf_pool->n_pend_reads--;
        buf_pool->stat.n_pages_read++;
        /* Release the X-latch or S-latch that was used during I/O. */
    } else if (io_type == BUF_IO_WRITE) {
        /* Mark the page clean, update flush stats, etc. */
    }
    ...
}
```
**Purpose**:
- Once the I/O actually finishes, InnoDB does final checks (checksums) and updates the buffer pool’s counters (`n_pend_reads`, etc.).
- This is also where it merges insert buffer entries if needed, or calls `recv_recover_page()` during crash recovery.

---

## 6. Flush Lists, Free Lists, LRU Lists

Throughout the code, you see references to:

- **`buf_pool->free`**: A list of blocks currently not used by any file page.  
- **`buf_pool->LRU`**: The main LRU list containing pages loaded in memory.  
- **`buf_pool->flush_list`**: Contains **dirty** pages that need to be written out eventually.  
- **`buf_pool->zip_clean`** (when `WITH_ZIP`): Contains **clean** compressed-only pages.

Functions like **`buf_LRU_add_block()`**, **`buf_LRU_free_block()`**, **`buf_flush_write_complete()`**, etc. manipulate these lists.

---

## 7. Compressed Pages Handling (`WITH_ZIP`)

Where you see `#ifdef WITH_ZIP`, InnoDB is dealing with **compressed** pages. Key points:

- A page can exist in memory in both a compressed form (`bpage->zip.data`) and, optionally, an uncompressed form (`block->frame`).  
- **`buf_block_t`** is used for uncompressed data; **`buf_page_t`** can be used for compressed-only.  
- Additional lists like **`buf_pool->zip_clean`** track compressed pages that aren’t dirty.  
- **`buf_buddy_alloc()`** is a small “buddy allocator” for compressed page frames.  
- **`buf_zip_decompress()`**: If a page is needed uncompressed, InnoDB decompresses it into a new block.

---

## 8. Debug and Validation Functions

- **`buf_validate()`**: Iterates over every chunk/block in the buffer pool and checks internal consistency:
  - Verifies that the LRU, free, and flush lists line up with each block’s state.
  - Ensures the counts match up (e.g., number of pages in flush list = number of dirty pages).
- **`buf_print()`** and **`buf_print_io()`**: Print debugging info: how many reads, writes, pages in LRU, free list length, pending I/O, etc.
- **`buf_get_latched_pages_number()`** (in debug builds): Counts how many pages are currently latched by any thread.

---

## 9. Shutdown and Cleanup

### **`buf_close()`** and **`buf_mem_free()`**
```c
UNIV_INTERN
void
buf_close(void) {
    ...
    hash_table_free(buf_pool->page_hash);
    buf_pool->page_hash = NULL;
#ifdef WITH_ZIP
    hash_table_free(buf_pool->zip_hash);
    buf_pool->zip_hash = NULL;
#endif
    ...
}

UNIV_INTERN
void
buf_mem_free(void) {
    if (buf_pool != NULL) {
        /* Free each chunk’s memory, then free buf_pool. */
        ...
        mem_free(buf_pool);
        buf_pool = NULL;
    }
}
```
**Purpose**:
- **`buf_close()`**: Prepares for shutting down InnoDB by freeing hash tables and other ephemeral structures.  
- **`buf_mem_free()`**: Actually releases the large memory chunks allocated for the buffer pool.

By the time these are called, the server is shutting down, and all pages should be flushed.

---

## Putting It All Together

- **Initialization**: `buf_pool_init()` sets up the buffer pool, chunks, free list, LRU list, and hash tables.  
- **Reading Pages**: `buf_page_get_gen()` is the main entry point to get a page from the buffer pool. If the page is not present, InnoDB schedules a read. Once in memory, the page is latched and pinned so the caller can safely access it.  
- **Writing Pages**: Dirty pages are put on the `flush_list`. A background thread (the flusher) eventually writes them out, calling `buf_page_io_complete()` when done.  
- **Compressed Pages**: `WITH_ZIP` code paths handle additional complexity for storing pages in compressed form.  
- **Cleanup**: On shutdown, InnoDB ensures all I/O is complete, flushes/writes dirty pages to disk, and then frees all memory used by the buffer pool.

This code is central to InnoDB’s performance and reliability because **every** data or index page read or written goes through the buffer pool. The design carefully balances concurrency (via mutexes, rw-locks, and I/O fixing) with correctness (checksums, flush ordering, lists) and performance (LRU, read-ahead, buddy allocator for compressed pages).

---

### **Key Takeaways**

1. **Data Structures**: Each in-memory page has a `buf_block_t` (for uncompressed) or `buf_page_t` (for compressed). They contain metadata like `io_fix`, `buf_fix_count`, LSN info, plus concurrency primitives (mutex/rw-lock).
2. **Hash Table**: `(space_id, page_no)` → pointer to the block. Speedy lookups to see if a page is in memory or not.
3. **LRU & Flush Lists**: Keep track of hot/cold pages and dirty pages. 
4. **Checksum Logic**: On each read, InnoDB verifies checksums to detect corruption.  
5. **Concurrency**: The global `buf_pool_mutex` is a “big” lock for critical operations, plus each page/block has its own lock for read/write concurrency.  
6. **Modular**: InnoDB supports compressing pages, partial reads, large memory allocations, dynamic resizing, etc. The code is sprinkled with `#ifdef`s for different build configurations (`UNIV_HOTBACKUP`, `WITH_ZIP`, etc.).

That’s the **high-level** explanation of what’s happening throughout `buf0buf.c`. If you dig deeper, you’ll see that many other files like `buf0lru.c`, `buf0flu.c`, `ibuf0ibuf.c`, and `srv0srv.c` complement or call these functions to implement the rest of the InnoDB memory and I/O management strategy.

# buf0buddy.c

**Short Answer**:  
This file (`buf0buddy.c`) implements a **“buddy” memory allocator** that InnoDB uses **specifically for compressed pages** (and their related control blocks). It manages allocations in powers of two, allowing InnoDB to efficiently split or merge memory blocks for storing compressed page data within the buffer pool.

---

## Why InnoDB Needs a Buddy Allocator for Compressed Pages

When InnoDB uses **page compression**, a page in the buffer pool can exist in both uncompressed (default 16 KB) and compressed form (e.g., 4 KB, 8 KB, etc.). Compressed pages:

1. **Free Up Buffer Pool Space**: If a page compresses well, InnoDB can store it in fewer bytes, effectively increasing how many pages can fit in memory.
2. **Have Variable Sizes**: Each compressed page might have a different compressed size (though still a power-of-two boundary internally, e.g., 2 KB, 4 KB, 8 KB, …).
3. **Must Be Dynamically Allocated**: InnoDB has to keep track of these compressed pages somewhere in memory, separate from the standard 16 KB “uncompressed” frames.

The **buddy allocator** in `buf0buddy.c` manages these varying-size memory blocks for compressed pages. It also tracks usage stats, can relocate pages to defragment memory, and merges “buddies” back into larger blocks when possible.

---

## Key Points in `buf0buddy.c`

1. **Buddy Blocks**:  
   - The code keeps a set of free lists, each for blocks of size \(2^n\) bytes (where \(n\) ranges in a certain range defined by `BUF_BUDDY_LOW` to `BUF_BUDDY_HIGH` or `PAGE_ZIP_MIN_SIZE` to `UNIV_PAGE_SIZE`).  
   - When InnoDB needs a compressed block of size 4 KB, it looks at the “free list” for 4 KB blocks. If empty, it splits a larger block (8 KB) into two 4 KB buddies, and so on.

2. **Allocation** (`buf_buddy_alloc_*`):  
   - **`buf_buddy_alloc_low()`** is a core function that tries to find a suitable free block from the buddy free lists. If needed, it will claim a free 16 KB frame from the buffer pool itself and split that into multiple smaller blocks.  
   - **`buf_buddy_alloc_zip()`** does the actual searching in free lists. If no block is available at a certain size, it attempts to split a bigger block from a higher-level free list.

3. **Freeing** (`buf_buddy_free_low()`):  
   - When a compressed page is no longer needed, the code frees that block back into the buddy system.  
   - The code attempts to **merge** (or “recombine”) it with its “buddy” block if that buddy is also free. Combining smaller blocks into bigger ones helps reduce fragmentation.

4. **Relocation** (`buf_buddy_relocate()`)  
   - If the buddy allocator needs to split or merge blocks, it might try to **move (relocate)** a compressed page or a `buf_page_t` control block to another spot.  
   - This relocation is guarded by checks to ensure the page can be safely moved (e.g., not pinned or busy with I/O).

5. **Statistics** (`buf_buddy_stat[]`)  
   - Tracks usage patterns: how many blocks are allocated, how many are freed, how often relocations happen, etc., to facilitate debugging or performance tuning.

6. **Integration with the Buffer Pool**  
   - The buddy allocator does **not** replace the entire buffer pool management. It operates **within** the buffer pool, dealing only with the memory segments used by compressed pages.  
   - You still have the main LRU list, free list, flush list, etc. for uncompressed pages. But any time InnoDB needs memory for a compressed page (or an associated metadata structure), it calls into this buddy system.

---

## How the Buddy System Works (High-Level)

1. **Powers-of-Two**:  
   Memory is managed in blocks of sizes that are powers-of-two. For example, if you have a 16 KB chunk, you can split it into two 8 KB “buddies,” each 8 KB can split into two 4 KB, and so forth.

2. **Split**:  
   If you request a 4 KB block but only have 8 KB free blocks, you split an 8 KB block into two 4 KB buddies, give one to the caller, and mark the other as free.

3. **Merge** (a.k.a. “Recombine”):  
   If a 4 KB buddy is freed, and its neighbor buddy (the other half of that 8 KB region) is also free, the code merges them back into an 8 KB block. Repeated merges help reduce fragmentation.

4. **Allocation from the LRU**:  
   If no sufficiently large free blocks exist in the buddy lists, it pulls a brand-new 16 KB frame from the buffer pool’s free list or by evicting something from the LRU.

---

## Summarizing the Purpose

**In short, `buf0buddy.c` implements an internal “binary buddy” memory allocator for storing compressed pages in InnoDB’s buffer pool.** It:

- Allows InnoDB to manage **variable-sized** compressed pages efficiently.
- Minimizes **fragmentation** via splitting and merging buddy blocks.
- Provides a mechanism to **relocate** compressed pages if needed.
- Integrates with the main InnoDB buffer pool for obtaining large (16 KB) blocks when smaller buddy blocks are exhausted.

By doing so, InnoDB can store compressed pages in smaller memory chunks while still having a systematic way to free and reuse that memory without excessive fragmentation.

# buf0flu.c

Below is a **high-level overview** of the **flush algorithm** used by InnoDB to write dirty pages from the buffer pool to disk, as implemented in **`buf0flu.c`**. The flush algorithm ensures that modified (dirty) pages in memory eventually get persisted to disk, while trying to balance performance, reduce I/O bursts, and avoid stalls.

---

## 1. What is “Flushing”?

In InnoDB, **dirty pages** are pages in the buffer pool that have been updated in memory but not yet written to disk. Eventually, these pages must be **flushed** (written) to the data files. The primary reasons:

1. **Durability**: If the server crashes, you don’t want to lose updates.
2. **Freeing Space**: The buffer pool needs free or “replaceable” pages for new data being read in. If too many pages remain dirty, you risk running out of reusable space in memory.
3. **Checkpointing**: InnoDB logs changes to the transaction log (redo log). The system uses a checkpoint mechanism that ensures no log record is older than the oldest unflushed page.

---

## 2. Data Structures Involved

1. **`flush_list`**: A doubly linked list of **all dirty pages** in the buffer pool, sorted roughly by the time (LSN) of their oldest modification (the oldest LSN that made that page dirty). The last element in the list is the page with the oldest changes (highest LSN).
2. **`LRU` (Least-Recently-Used) list**: All pages in the buffer pool (both clean and dirty), sorted by recency of use. The tail end has the least recently used pages, which are eviction candidates.
3. **Per-Page Fields**:
   - **`oldest_modification`**: The LSN of the oldest log record that modified this page. If it’s `0`, the page is clean.
   - **`buf_fix_count`**: The number of threads currently “pinning” (fixing) the page in memory.
   - **`io_fix`**: Indicates if a page is being read or written (`BUF_IO_READ`, `BUF_IO_WRITE`), or none (`BUF_IO_NONE`).

---

## 3. Types of Flush Operations

There are primarily **two flush types** in the code:

1. **`BUF_FLUSH_LIST`**:  
   - This flush targets pages from the **flush_list** (i.e., all dirty pages).  
   - Typically used to keep the overall number of dirty pages at a manageable level (to avoid big bursts of I/O later) and to bound how far the redo log can grow.

2. **`BUF_FLUSH_LRU`**:  
   - This flush targets pages from the **LRU** tail to ensure there are enough free/replaceable pages in memory for new data.  
   - It tries to flush pages from the “cold” end of the LRU so that they can be quickly evicted if needed.

A thread might call **`buf_flush_batch(BUF_FLUSH_LRU, …)`** or **`buf_flush_batch(BUF_FLUSH_LIST, …)`** depending on the situation.

---

## 4. Highlights of the Algorithm

### 4.1 Checking If a Page is Ready to Flush
- **`buf_flush_ready_for_flush(bpage, flush_type)`** checks if:
  1. **`bpage->oldest_modification != 0`** (the page is dirty).
  2. **`io_fix`** is `BUF_IO_NONE` (no ongoing I/O on that page).
  3. For LRU flushes, **`bpage->buf_fix_count == 0`** (nobody is using the page).  
  If these conditions are met, the page can be flushed immediately.

### 4.2 Removing a Page from the Flush List
- **`buf_flush_remove(bpage)`**: Once a page is successfully written, we mark it clean (`oldest_modification = 0`) and remove it from `flush_list`. If it’s compressed (WITH_ZIP), we also convert its state from `ZIP_DIRTY` to `ZIP_PAGE`.

### 4.3 The “Flush Batch” (`buf_flush_batch()`)
- This is the top-level function to flush a group of dirty pages:
  1. Prevent multiple flushes of the same type from running at once (if a flush is in progress, it returns `ULINT_UNDEFINED`).
  2. Depending on `flush_type`:
     - **LRU flush**: Start from the tail of the LRU list, look for pages ready to flush.
     - **Flush list**: Start from the oldest-dirty end of `flush_list`.
  3. For each flushable page, the code tries to also flush “neighbors” (pages with consecutive offsets) for efficiency.
  4. Schedules the writes asynchronously (or via the doublewrite buffer if configured).

### 4.4 Doublewrite Buffer
- If **`srv_use_doublewrite_buf`** is on, the page is first written to a special **doublewrite** memory buffer, then from there to disk. This avoids partial-page writes and helps ensure crash recovery correctness.
- **`buf_flush_post_to_doublewrite_buf()`** adds the page to the doublewrite buffer array. If that buffer is full, **`buf_flush_buffered_writes()`** is called, which physically writes the pages from the doublewrite buffer to disk, then writes them again to their final locations, and finally flushes the OS cache with `fil_flush()`.

### 4.5 Syncing the Log
- Before writing the page, InnoDB ensures that the log up to the page’s **`newest_modification`** LSN is flushed to disk.  
- This ensures **WAL (Write-Ahead Logging)**: the redo log for changes must reach disk before the data page hits disk.

---

## 5. Controlling Flush Frequency

### 5.1 LRU Margin Flushing
- **`buf_flush_free_margin()`**: If the buffer pool is low on free pages, it flushes pages from the LRU tail to free them up more quickly. This tries to maintain `BUF_FLUSH_FREE_BLOCK_MARGIN` free pages.

### 5.2 Redo Log Growth Heuristic
- The code collects **statistics** in `buf_flush_stat_update()` about:
  - How fast the **redo log** is growing (`redo_avg`).
  - How many pages are being flushed by normal LRU flushes (`lru_flush_avg`).
- **`buf_flush_get_desired_flush_rate()`** calculates how many dirty pages should be flushed per second to avoid large bursts. It uses the formula roughly:  
  \[
      \text{desired} = \text{(ratio of redo generated)} - \text{(pages already being flushed by LRU)}
  \]

In effect, if the system is generating redo log very quickly but not flushing enough from the flush list, InnoDB schedules additional flush list writes to avoid big IO spikes later.

---

## 6. Typical Flow of a Flush

Here’s a simplified, typical flow when a page is to be flushed via the **flush list**:

1. A background thread (like the “page cleaner” thread in modern InnoDB) calls `buf_flush_batch(BUF_FLUSH_LIST, min_n, lsn_limit)`:
   1. Checks if another flush list batch is active. If yes, return.
   2. Iterates the flush list from newest to oldest (or vice versa).  
   3. For each eligible dirty page:
      - Acquire the page mutex, check if it’s not pinned by other threads (`buf_fix_count = 0`) and `io_fix = BUF_IO_NONE`.
      - Mark `io_fix = BUF_IO_WRITE`.
      - Release the buffer pool mutex and the page mutex (to avoid holding them during actual I/O).
      - Call `buf_flush_write_block_low()`, which either writes directly or uses the doublewrite buffer.
2. After scheduling those writes, it calls `buf_flush_buffered_writes()` if using doublewrite, which ensures those writes actually make it to disk.
3. Once the I/O completes, the function `buf_page_io_complete()` marks the page clean and removes it from the flush_list.

For an **LRU flush**, the principle is similar, except the algorithm iterates from the **tail** of the LRU list, picking dirty pages that are not actively in use, flushing them to free up space in memory.

---

## 7. Summary of Key Functions

- **`buf_flush_insert_into_flush_list()`**: Adds a newly-dirtied page to the flush_list (at the head, since it’s sorted by `oldest_modification` descending).
- **`buf_flush_remove()`**: Marks a page as clean and removes it from flush_list once it’s written.
- **`buf_flush_batch(flush_type, min_n, lsn_limit)`**: The main function to flush up to `min_n` pages of a given type (LRU or flush_list).
- **`buf_flush_write_block_low()`**: Actually posts an asynchronous write request to the disk (directly or via doublewrite).
- **`buf_flush_buffered_writes()`**: If doublewrite is used, flushes the pages from the doublewrite buffer to disk, then from the buffer to final positions.
- **`buf_flush_wait_batch_end(flush_type)`**: Waits until the flush batch of a certain type finishes (i.e., all writes have completed).

---

## 8. Why This Matters

- **Minimize I/O Spikes**: By gradually flushing pages (rather than waiting until dirty pages build up excessively), InnoDB avoids big bursts of disk activity.
- **Keep Enough Free Pages**: LRU flushing ensures the buffer pool won’t stall for lack of available frames.
- **Ensure Write-Ahead Logging**: Data on disk is always in sync with or behind the redo log (no partial pages or out-of-order writes).
- **Crash Recovery**: The doublewrite buffer + consistent flushing approach ensures data-file pages and logs remain consistent if a crash happens mid-write.

---

### Final Takeaway

The **flush algorithm** in `buf0flu.c` orchestrates how InnoDB writes dirty pages from memory to disk in a manner that balances performance (by doing incremental writes, batching neighbors, and relying on heuristics for controlling flush rate) and correctness (through doublewrite buffer, checksums, and WAL). This is a **core** part of InnoDB’s reliability and performance strategy.

#

Below is a **high-level explanation** of how **InnoDB’s buffer replacement algorithm** (the **LRU—Least Recently Used** algorithm) works, based on the code in **`buf0lru.c`**. This file contains the logic for:

1. **Arranging pages in the buffer pool** into a primary LRU list.
2. **Choosing victim pages** (i.e., evicting pages from memory) to free space for new pages.
3. Splitting pages in the LRU list into “young” and “old” segments to optimize for different access patterns.
4. Integrating with compressed page logic (`WITH_ZIP`).

The key data structures here are:

- The **LRU list** (`buf_pool->LRU`): All pages in the buffer pool (except those on the free list) are linked here in approximate order of most-recently accessed (the front) to least-recently accessed (the tail).
- The **unzip_LRU list** (`buf_pool->unzip_LRU` when `WITH_ZIP` is enabled): Tracks pages that exist in **both** compressed and uncompressed form, so we can quickly discard the uncompressed copy if memory is tight.
- A **free list** (`buf_pool->free`): Contains pages not currently holding any data (i.e., truly free). When we need a new page, we take from here if available.

The functions in `buf0lru.c` manage how pages move among these lists, and how InnoDB chooses which pages to evict or free when space is needed.

---

# buf0lru.c

## 1. LRU List Mechanics

### 1.1 Adding Pages to the LRU List

- **`buf_LRU_add_block()`**: When a new page (either read in from disk or created) gets placed into the buffer pool, it’s added to the **front** (most-recently-used side) of the LRU list unless directed otherwise.  
- **Old vs. New Blocks**: InnoDB splits the LRU into “new” (the front portion) and “old” (the tail portion). The boundary between these segments is tracked by **`buf_pool->LRU_old`**, and the approximate size of the “old” segment is determined by `buf_LRU_old_ratio`.  

### 1.2 Making a Page “Young”

- **`buf_LRU_make_block_young()`**: If a page is accessed again after some time, InnoDB can move it from the “old” part of the LRU list to the “young” side (the front of the list). This is effectively how “recent usage” is recognized.

### 1.3 Making a Page “Old”

- **`buf_LRU_make_block_old()`**: Sometimes InnoDB explicitly moves a page to the tail of the list—especially if we know it’s not going to be reused soon. This helps free it up sooner.

---

## 2. Splitting the LRU Into New and Old Segments

InnoDB tries to preserve a certain fraction of the LRU list as “old” blocks (the tail). Specifically:

```c
buf_LRU_old_ratio = old_pct * BUF_LRU_OLD_RATIO_DIV / 100;
```

The code ensures that about `old_pct%` of the LRU list is treated as “old.” This is an optimization to avoid “index scans” or large table scans from polluting the entire buffer pool with pages that might not be reused soon.  

**Key functions**:

- **`buf_LRU_old_init()`**: Called once the LRU list reaches a minimum length. It marks all pages as old, then re-adjusts to find a proper position for the boundary.  
- **`buf_LRU_old_adjust_len()`**: Continuously adjusts the pointer `buf_pool->LRU_old` as new pages come in or pages move to the front, keeping the ratio of old/new stable within a tolerance (`BUF_LRU_OLD_TOLERANCE`).

---

## 3. Eviction (Finding a Victim)

### 3.1 Searching from the LRU Tail

When InnoDB needs a free page (e.g., a new page is read from disk and no “free list” pages are available), it tries to **evict** from the tail of the LRU list. The tail (the “old” end) should hold the least-recently-used pages.  

**Core function**:
- **`buf_LRU_search_and_free_block()`**: Repeatedly looks at the LRU tail to find a page that’s **not** pinned (i.e., `buf_fix_count == 0`) and **not** being I/O-fixed (`io_fix == BUF_IO_NONE`), and is **clean** if we’re discarding it entirely.  
- If a page is **dirty**, InnoDB must flush it first (i.e., write it to disk) before eviction, or choose a different page. That’s integrated with the flush logic in `buf0flu.c`.

### 3.2 Freed or Compressed Page Cases

- If **`WITH_ZIP`** is enabled, InnoDB might keep pages in both compressed and uncompressed form. If the page is dirty, it must remain in memory (or be written to disk). But if the uncompressed copy is not needed, InnoDB can keep only the compressed copy.  
- **`buf_LRU_free_block()`**: Attempts to remove the page from the LRU list, remove its entry from the page hash, and put the block on the free list. If the page is compressed-only, it might only discard the uncompressed buffer.

### 3.3 The Unzip LRU List

- If a page is in **both** compressed and uncompressed form, the uncompressed portion can be freed while keeping the compressed data. This is tracked in **`unzip_LRU`**.  
- **`buf_LRU_free_from_unzip_LRU_list()`** tries to discard the uncompressed copy only.  

---

## 4. The Free List

- After a page is evicted from the LRU (or if it was never used), it’s put on the **`buf_pool->free`** list.  
- When InnoDB needs a free page, it first checks `free`. If that’s empty, it tries the LRU tail.  

**Important**: The code tries to ensure there’s always a margin of free pages so that new pages can be brought in without stalling.

---

## 5. Handling Compressed Pages (`WITH_ZIP`)

You’ll see numerous `#ifdef WITH_ZIP` sections. Key points:

1. **`unzip_LRU`**: Pages that exist in memory both compressed and uncompressed.  
2. **`buf_LRU_evict_from_unzip_LRU()`** decides whether to evict just the uncompressed copy or the entire page.  
3. **`buf_buddy_alloc()`** (from `buf0buddy.c`) is used for allocating memory for compressed pages. If a block is “relocatable,” InnoDB might move its data in memory.  

---

## 6. “Young” vs. “Old” Access Tuning

- **`buf_LRU_old_threshold_ms`**: If a page’s first access was very recent (under some threshold), InnoDB might consider it part of a large scan and keep it in the “old” segment.  
- This is a **heuristic** to avoid large table/index scans from displacing more important pages.

---

## 7. Additional Notable Functions

1. **`buf_LRU_drop_page_hash_for_tablespace()`**: When dropping or discarding a tablespace, InnoDB attempts to remove all pages of that tablespace from the buffer pool (and from the page hash).
2. **`buf_LRU_invalidate_tablespace()`**: Another step to ensure a dropped tablespace’s pages are all removed from memory (and possibly flush them if dirty).
3. **`buf_LRU_try_free_flushed_blocks()`**: If some pages have just been flushed (written to disk) from the tail, see if they can be quickly moved to the free list before they get re-dirtied.

---

## 8. Statistics and Debug

- **`buf_LRU_stat_update()`**: Tracks how many I/O operations vs. page decompressions happen, used to decide eviction policies (whether the workload is more “disk-bound” or “CPU-bound”).
- **`buf_LRU_validate()`**, **`buf_LRU_print()`**: Debug routines to ensure the list is consistent and/or print out the LRU chain for analysis.

---

## 9. Summary of the Buffer Replacement Algorithm

1. **LRU List**  
   - Pages are in a big linked list from newest (head) to oldest (tail).  
   - An internal pointer (`LRU_old`) separates “young” from “old” pages.  

2. **Eviction**  
   - InnoDB evicts pages primarily from the tail (“old” side) of the LRU list.  
   - If a page is pinned, dirty, or being I/O-fixed, it can’t be evicted immediately.  
   - If the page is compressed and uncompressed, InnoDB can drop just the uncompressed copy (if the page is dirty, we still keep the compressed version).  

3. **Free List**  
   - Freed pages go to the `free` list for fast reuse. If `free` is empty, InnoDB does more evictions from LRU.  

4. **Unzip LRU**  
   - Sub-list to handle compressed+uncompressed pages. We can either discard just the uncompressed part or the entire page.  

5. **Adjusting “Old” vs. “Young”**  
   - `buf_LRU_old_ratio` ensures a certain fraction of the buffer pool is deemed “old.”  
   - Large scans that quickly access a lot of new pages get placed into the “old” segment so they don’t displace truly hot pages.  

This algorithm is designed to **balance performance** by keeping frequently accessed pages in memory (near the LRU front), while efficiently **evicting** pages that are either unlikely to be accessed again or can’t remain (due to memory pressure). The interplay of old/new segments and the ability to keep only compressed copies helps InnoDB handle different workloads efficiently.

# buf0rea.c

Below is a **high-level explanation** of the code in **`buf0rea.c`**, which handles **reading** pages into the InnoDB buffer pool. This includes:

- The primary function to read a single page into the buffer pool.
- Various **read-ahead** mechanisms (e.g. linear read-ahead).
- Special reads for **ibuf merges** and **recovery**.

---

## 1. Overview of Reading Into the Buffer Pool

The fundamental goal is to take a page on disk (identified by **space** and **page offset**) and load it into an available buffer pool frame. InnoDB can do this **synchronously** (the calling thread waits for I/O to complete) or **asynchronously** (post an I/O and let an I/O handler thread finish the work). The code must handle:

1. Checking if the page is **already** in the buffer pool. If so, no read needed.
2. Creating a **victim** buffer frame if none is free, potentially flushing or evicting a page (see `buf0lru.c` and `buf0flu.c`).
3. Handling special cases:
   - Pages in the **doublewrite buffer** (never read from disk into the normal pool).
   - Ibuf (insert buffer) pages.
   - Recovery-time reads.
4. **Read-ahead** heuristics: If InnoDB detects a sequential or near-sequential pattern, it preemptively issues multiple reads.

---

## 2. Core Reading Function: `buf_read_page_low()`

```c
UNIV_STATIC
ulint
buf_read_page_low(
    ulint*   err,
    ibool    sync,
    ulint    mode,
    ulint    space,
    ulint    zip_size,
    ibool    unzip,
    ib_int64_t tablespace_version,
    ulint    offset
)
```

### What it Does

1. **Checks** if the page is in the **doublewrite buffer** (part of the system tablespace). If so, returns without reading.
2. **Allocates** or “init for read” (`buf_page_init_for_read()`) a control block in the buffer pool for this `(space, offset)` if it’s not already there:
   - If the tablespace is deleted or being dropped, it sets `*err = DB_TABLESPACE_DELETED`.
   - Otherwise, it sets `io_fix = BUF_IO_READ` on the page object.
3. **Issues** an I/O via `fil_io()`:
   - `sync == TRUE` → synchronous I/O. The function waits until the read completes, then calls `buf_page_io_complete()`.
   - `sync == FALSE` → asynchronous I/O. It posts the read to the OS, and an I/O handler thread will complete it later.

### Return Value

- Returns `1` if it actually queued (or performed) a read.
- Returns `0` if:
  - The page was **already** in the buffer pool.
  - The page is part of the doublewrite buffer (no read needed).
  - The tablespace does not exist or is being dropped.

---

## 3. Higher-Level `buf_read_page()`

```c
UNIV_INTERN
ibool
buf_read_page(ulint space, ulint zip_size, ulint offset)
{
    ...
    count = buf_read_page_low(..., TRUE, ...); // synchronous read
    ...
    return(count > 0);
}
```

- **Always** does a **synchronous** read (the calling thread waits).
- Returns `TRUE` if the page read was actually issued, `FALSE` otherwise.
- Increments counters like `srv_buf_pool_reads`.
- Calls `buf_flush_free_margin()` afterward in case we need to free LRU pages.

This is typically used in operations that cannot tolerate waiting for an asynchronous callback, e.g., certain “critical” reads.

---

## 4. Linear Read-Ahead: `buf_read_ahead_linear()`

```c
UNIV_INTERN
ulint
buf_read_ahead_linear(
    ulint space,
    ulint zip_size,
    ulint offset
)
```

### Rationale

When InnoDB notices a page has been read whose offset is on the **border** of a certain region (`BUF_READ_AHEAD_LINEAR_AREA` blocks), it checks if **most** pages in that region were accessed in ascending or descending order. If so, it predicts the user might read the **next** region soon, so it issues read requests for those pages preemptively. 

### Key Steps

1. **Check** if `offset` is at the boundary (e.g., `offset == low` or `offset == high - 1`) in a `BUF_READ_AHEAD_LINEAR_AREA`.
2. **Verify** that many pages in that area have actually been accessed (using `buf_page_is_accessed()`—which tracks the “first” access time).
3. **Peek** at the page’s successor or predecessor (`fil_page_get_next()` / `fil_page_get_prev()` in the page header).
4. If conditions are met (the new region is within space bounds, etc.), do an **asynchronous** read of the entire region in that next chunk.

The code uses a small heuristic (`fail_count` vs. `threshold`) to allow some out-of-order page accesses before deciding the pattern is not truly linear.

This read-ahead is triggered typically **the moment** a page is first accessed in the buffer pool. In practice, this can significantly reduce I/O latency if the user is scanning through pages in ascending/descending offset order.

---

## 5. Insert Buffer Merge Reads: `buf_read_ibuf_merge_pages()`

```c
UNIV_INTERN
void
buf_read_ibuf_merge_pages(
    ibool       sync,
    const ulint* space_ids,
    const ib_int64_t* space_versions,
    const ulint* page_nos,
    ulint       n_stored
)
```

### Purpose

- During **insert buffer** merges, InnoDB may need to read multiple leaf pages from indexes to apply the buffered changes (“merge them”). 
- This function batches those reads. It’s similar to read-ahead: it issues multiple read requests to bring the needed pages into the buffer pool.

### Details

- It loops over each `(space_id, page_no)` pair, calling `buf_read_page_low()` with a mode that may be synchronous for the last page or asynchronous for earlier pages.
- If the tablespace is gone (`DB_TABLESPACE_DELETED`), it calls `ibuf_merge_or_delete_for_page()` to discard any pending changes for those pages.

---

## 6. Recovery Reads: `buf_read_recv_pages()`

```c
UNIV_INTERN
void
buf_read_recv_pages(
    ibool   sync,
    ulint   space,
    ulint   zip_size,
    const ulint* page_nos,
    ulint   n_stored
)
```

### Purpose

During **crash recovery**, InnoDB must read pages that the redo log indicates need changes. This function:

- Does repeated calls to `buf_read_page_low()` with a possible wait if there are too many pending reads.
- Possibly does a synchronous read on the last page to ensure it’s definitely in memory.

Since recovery can be single-threaded or at least partially so, the function tries not to flood the buffer pool with too many pending reads at once.

---

## 7. Supporting Details

- **`BUF_READ_ANY_PAGE`** vs. `BUF_READ_IBUF_PAGES_ONLY`: 
  - `BUF_READ_ANY_PAGE` indicates normal read-ahead or normal read. 
  - `BUF_READ_IBUF_PAGES_ONLY` restricts to only insert buffer pages.  
- **“Simulated AIO”**: On some platforms, InnoDB might not have true OS asynchronous I/O. Instead, it uses a simulated approach with dedicated I/O threads. The code often calls `os_aio_simulated_put_read_threads_to_sleep()` or `os_aio_simulated_wake_handler_threads()` to coordinate these tasks.
- **Check `buf_flush_free_margin()`** after a series of reads to ensure there is enough free buffer space (pages at the tail of LRU can be moved or evicted).

---

## 8. Summary

`buf0rea.c` is dedicated to **reading** pages into the InnoDB buffer pool. Its responsibilities include:

- **Single-page** read logic (`buf_read_page()`, `buf_read_page_low()`).
- **Read-ahead** heuristics, especially linear read-ahead (`buf_read_ahead_linear()`).
- **Special-case read** routines for insert buffer merges (`buf_read_ibuf_merge_pages()`) and crash recovery (`buf_read_recv_pages()`).
- Integrating with the rest of InnoDB’s concurrency and memory management logic (LRU flushing, doublewrite buffer checks, etc.).

Together with the code in `buf0buf.c`, `buf0lru.c`, and `buf0flu.c`, it forms the complete **buffer pool** system that orchestrates how pages are fetched from disk, cached in memory, and eventually freed or evicted.