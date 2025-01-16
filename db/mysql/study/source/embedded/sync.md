# sync0arr.c

Below is an explanation of the main ideas behind **sync0arr.c** (the “wait array”) in InnoDB and how it differs from using a simple/normal C array.

---

## Overview: Why InnoDB has a “Wait Array”

InnoDB implements its own synchronization primitives (e.g., custom mutexes and read-write locks) instead of relying purely on the operating system’s locks. Historically, InnoDB used a global wait array of fixed-sized “cells” (each with an OS event) to manage threads that were waiting for locks. This design was chosen because, in older operating systems, creating a very large number of OS-level events (one per lock) was prohibitively expensive. By using a single global wait array, InnoDB could reuse a small pool of OS events when threads needed to wait.  

Although this changed somewhat in later InnoDB versions (now the OS event is often embedded directly in the mutex/rw-lock object itself), **the wait array is still kept for diagnostics** (to track waiting threads, avoid infinite waits, detect potential deadlocks, etc.).

---

## How the Wait Array Differs from a Normal Array

1. **Each “cell” is a specialized data structure, not just a simple element.**  
   In a normal C array of `int` or `char`, each element is just a data field. In contrast, each element in the wait array (`sync_cell_struct`) is a small record with:
   - A pointer to the synchronization object the thread is waiting on (`wait_object`).
   - The type of lock request (e.g., exclusive lock, shared lock, etc.).
   - OS-specific event or signal-count information (`signal_count`).
   - Diagnostics: file name, line number, thread ID, and timestamps.
   - Flags to indicate if the thread is currently waiting (`waiting`).

2. **The array is protected by a mutex/OS mutex.**  
   Because multiple threads will be adding and removing themselves from the wait array, there must be a *global lock* (either an OS mutex or an InnoDB mutex) that protects these operations. The functions `sync_array_enter()` and `sync_array_exit()` acquire and release this protecting lock.

3. **Cells are “reserved” and “freed” instead of being written arbitrarily.**  
   A normal array might be used with an index operator (`arr[i] = value;`). In contrast, here you do:
   - **Reserve a cell**: find a free cell, mark it as in use, store the waiting info.
   - **Wait** on that cell’s associated event.
   - **Free** the cell after the wait completes.  

   The code maintains a count of how many cells are currently reserved (`n_reserved`) and a total of how many cells (`n_cells`) exist.

4. **It embeds concurrency logic and debugging / deadlock detection.**  
   - `sync_array_detect_deadlock()` traverses from one waiting cell to the thread that owns the blocking lock, recursively, to check if there is a cycle (deadlock).
   - Timestamps and counters in each cell help detect threads that have waited too long (`sync_array_print_long_waits()`), produce debug output, and log state to help with diagnosing concurrency problems.

5. **Not truly a single, contiguous memory region for user data.**  
   Although it is an array in memory (`arr->array = ut_malloc(sizeof(sync_cell_t) * n_cells)`), each slot is not user data but a mini-structure for concurrency:

   ```cpp
   struct sync_cell_struct {
       void*         wait_object;       // The mutex or rw-lock
       ulint         request_type;      // SYNC_MUTEX, RW_LOCK_EX, etc.
       os_thread_id_t thread;           // OS thread ID
       ibool         waiting;           // Has thread called event_wait?
       time_t        reservation_time;  // For debugging waits
       ...
   };
   ```

---

## Key Functions in `sync0arr.c`

1. **`sync_array_create()`**  
   Allocates the wait array structure and either creates an OS mutex or an InnoDB mutex to protect the array.  

2. **`sync_array_reserve_cell()`**  
   - Finds a free cell (`cell->wait_object == NULL`).
   - Stores the caller’s object pointer (`object`), the request type (`type`), the file/line (for debugging), and marks the cell as reserved.  
   - Resets the event (`os_event_reset`) so that the waiting thread will properly block next time it calls `os_event_wait()`.

3. **`sync_array_wait_event()`**  
   - Waits for the OS event belonging to that cell (which was set up in `sync_array_reserve_cell()`).  
   - Once the event is signaled, the function frees the cell (`sync_array_free_cell()`) so it can be reused by others.

4. **`sync_arr_wake_threads_if_sema_free()`**  
   - Periodically checks if a synchronization object is available now (e.g., the mutex got unlocked).  
   - If yes, it sets the associated OS event to wake any thread waiting for that cell.  
   - This function helps recover from missed signals or anomalies in the normal wake-up logic.

5. **`sync_array_print_long_waits()`**  
   - Scans all reserved cells.
   - If a thread has been waiting longer than a certain threshold, prints a warning or triggers debug output (the “InnoDB Monitor”).

---

## Putting It All Together

In short, **the wait array**:

- Maintains a global list of threads that are waiting on InnoDB synchronization objects (mutexes, rw-locks).
- Each cell in the array holds:
  - Which object is being waited on.
  - The type of wait (shared vs. exclusive lock, etc.).
  - Thread ID, signal counts, and debugging info.
- A separate global protecting mutex manages concurrency on the array itself.
- Diagnostic routines help detect deadlocks and threads stuck waiting too long.

Compared to a normal array, it’s **not** just a place to store raw data. Instead, each array entry is a mini control block managing wait-state information. The “cells” get reserved and freed dynamically, and there is a lot of logic around each cell to handle concurrency, signaling, deadlock detection, and debugging.

This design (global array of wait cells) is largely historical and was originally meant to avoid creating too many OS events. Even though modern systems can handle many events, **InnoDB keeps this array** around for its advanced diagnostic, fail-safe, and debugging capabilities.

# sync0rw.c

Below is an explanation of the **read-write lock** (`rw_lock_t`) code in **sync0rw.c**. It builds on ideas from the wait array (`sync0arr.c`), but focuses on **how InnoDB implements and manages read-write locks** with atomics (when available), spin locks, OS waits, and debugging/troubleshooting helpers.

---

## Purpose of This Code

1. **Implements InnoDB’s custom read-write (RW) locks**:
   - RW locks let multiple readers hold the lock **or** a single writer hold it exclusively.
   - InnoDB needs fine-tuned performance on top of multiple OSes.  
   
2. **Uses both spin-waits and OS-level waits**:
   - *Spin-wait*: briefly busy-waits (loops checking a condition) hoping the lock becomes free soon (fast if locked briefly).
   - *OS wait*: if lock still isn’t free after spinning, the thread sleeps on an OS event until awakened.

3. **Diagnostic and debugging features**:  
   - InnoDB can track which thread holds the lock, who’s waiting, the file/line where it was acquired, etc.
   - Supports recursion for exclusive locks (the same thread can lock multiple times).  
   - Maintains a global list of all RW locks, each with debugging info to help detect deadlocks or long lock-hold times.

---

## Key Structures

### 1. `rw_lock_t`
Core structure of a read-write lock:
```c
struct rw_lock_struct {
    ib_uint32_t    lock_word;      // The main lock state
    ibool          waiters;        // Are threads waiting on this lock?
    ibool          recursive;      // TRUE => allows recursion by same thread
    os_thread_id_t writer_thread;  // Which thread holds the exclusive lock
    os_event_t     event;          // Event for normal (S or X) waiters
    os_event_t     wait_ex_event;  // Event for "waiting X lock" (WAIT_EX)
    // ... plus various debug fields ...
};
```
The `lock_word` is the heart of the lock:
- **`X_LOCK_DECR`** is a constant (usually 256) that indicates an exclusive lock decrement. 
- The lock_word can be:
  - **X_LOCK_DECR**: lock is unlocked (i.e. 256 if X_LOCK_DECR=256).
  - Between **0** and **X_LOCK_DECR**: lock is read-locked. The actual number of readers = `X_LOCK_DECR - lock_word`.
  - Exactly **0**: lock is fully exclusive-locked (non-recursive).
  - Less than **0**: indicates either read-locked with a waiting writer or a (possibly recursive) exclusive lock.  

Reading or updating `lock_word` is done atomically, so the entire state is known in a single 32-bit read.

### 2. `rw_lock_list` and `rw_lock_list_mutex`
A global, linked list of all RW locks—used primarily for debugging and printing out lock states. Protected by `rw_lock_list_mutex`.

```c
UNIV_INTERN rw_lock_list_t   rw_lock_list;
UNIV_INTERN mutex_t          rw_lock_list_mutex;
```

---

## How RW Locking Works (High Level)

1. **Shared (reader) lock**:  
   - If no writer or waiting writer (i.e., `lock_word > 0`), the thread just decrements `lock_word` by **1** atomically. If successful, it becomes another reader.  
   - If a writer is active or waiting (`lock_word <= 0`), it spins for a while (configurable with `SYNC_SPIN_ROUNDS`) to see if the lock frees up. If not, it **OS-waits**.  

2. **Exclusive (writer) lock**:  
   - The thread tries to decrement `lock_word` by **X_LOCK_DECR** (e.g., 256) in an atomic step.  
   - If successful, it means this thread is the next writer or the current writer (for recursion). It must then wait until all readers have exited (i.e., `lock_word >= 0`) before proceeding.  
   - If that fails (someone else is writing, or a waiting writer is already queued), it also spins, then OS-waits.

3. **Recursive exclusive locks**:
   - The same thread can acquire the same lock multiple times. The code checks `writer_thread` and `recursive` flags to decide if the lock can be incrementally re-locked.

4. **Spin + OS Wait**:
   - Typically, the thread tries to acquire the lock in a tight loop up to `SYNC_SPIN_ROUNDS`.  
   - If that fails, it puts itself into the **global wait array** (`sync_array_reserve_cell(...)`) and goes to sleep via an OS event. Another thread will wake it when the lock is free (`os_event_set`).

---

## Major Functions

### 1. **`rw_lock_create_func()`**  
Initializes a new lock in memory:
```c
void
rw_lock_create_func(
    rw_lock_t* lock,
    ...
    const char* cfile_name,
    ulint cline)
{
    // Possibly create a small InnoDB or OS-level mutex
    // if INNODB_RW_LOCKS_USE_ATOMICS is undefined.

    lock->lock_word = X_LOCK_DECR; // unlocked
    lock->waiters   = 0;
    lock->recursive = FALSE;
    ...
    lock->event         = os_event_create(NULL);
    lock->wait_ex_event = os_event_create(NULL);

    // Add lock to the global list (for debugging)
    mutex_enter(&rw_lock_list_mutex);
    UT_LIST_ADD_FIRST(list, rw_lock_list, lock);
    mutex_exit(&rw_lock_list_mutex);
}
```
- If the code supports “atomic” RW locks (`INNODB_RW_LOCKS_USE_ATOMICS`), the lock acquires or releases using CPU atomic instructions. Otherwise, an internal `mutex` is used.

### 2. **`rw_lock_free()`**  
Frees resources (the OS events, and removes from the global list). The lock must be in an unlocked state first:
```c
void
rw_lock_free(rw_lock_t* lock)
{
    // Check it’s unlocked
    ut_a(lock->lock_word == X_LOCK_DECR);

    // Free associated OS resources
    os_event_free(lock->event);
    os_event_free(lock->wait_ex_event);

    // Remove from global list
    ...
}
```

### 3. **Locking Routines**

#### a) Shared Lock: `rw_lock_s_lock_spin()`
```c
void
rw_lock_s_lock_spin(rw_lock_t* lock, ulint pass, const char* file_name, ulint line)
{
    // Spin a while to see if lock->lock_word becomes > 0
    // If it doesn't, reserve a wait cell in sync_array, set waiters=1, sleep
    // When awakened, try again
}
```
1. **Spin** while `lock_word <= 0`.  
2. If it never becomes positive, call `sync_array_reserve_cell()` and go to sleep on `lock->event`.

#### b) Exclusive Lock: `rw_lock_x_lock_func()`
```c
void
rw_lock_x_lock_func(rw_lock_t* lock, ulint pass, const char* file_name, ulint line)
{
    // Similar spin approach, but tries to decrement lock_word by X_LOCK_DECR.
    // If successful, we are 'next writer' and must wait for any readers to drain out.
    // If not, we eventually wait in sync_array for the lock->event or lock->wait_ex_event.
}
```
- If the same thread is already the writer (recursive), it just increments the recursion count.

### 4. **`rw_lock_x_lock_wait()`**  
After decrementing `lock_word` by `X_LOCK_DECR`, you might have to wait for remaining readers to exit (while `lock_word < 0`). This function implements that wait if needed.

### 5. **Debugging & Checking** (when `UNIV_SYNC_DEBUG` is enabled):
- **`rw_lock_add_debug_info()`** / **`rw_lock_remove_debug_info()`**:  
  Add/remove a record of which thread locked the RW lock, in which mode (S or X), the file/line, etc. Helps detect deadlocks or see who holds a lock.

- **`rw_lock_print()`**, **`rw_lock_list_print_info()`**:  
  Dump debug info for a single lock or all known locks.

---

## How It Differs From a “Normal” C Implementation

1. **Atomic `lock_word`**:  
   In a typical read-write lock in plain C, you might protect shared counters with a single pthread mutex or multiple condition variables. Here, **InnoDB uses a single atomic integer** (`lock_word`) to represent the entire lock state, plus explicit OS events for waiting.

2. **Global Wait Array Integration**:  
   Instead of creating a per-lock condition variable, InnoDB historically used a **global wait array** (`sync0arr.c`) to manage waiting threads. Routines like:
   ```c
   sync_array_reserve_cell(sync_primary_wait_array, lock, RW_LOCK_EX, file, line, &index);
   sync_array_wait_event(sync_primary_wait_array, index);
   ```
   This central structure tracks all waiting threads, each cell having its own OS event.  

3. **Spinning**:  
   InnoDB explicitly implements spin loops in C (with possible random backoff, `ut_delay`, etc.), rather than letting pthread library do the spin/wait.

4. **Recursive Exclusive Locks**:  
   Because of InnoDB’s internal concurrency design, the same thread might need to re-acquire the same lock multiple times. The code manages recursion by tracking `writer_thread` and a `recursive` flag in the lock.

5. **Extensive Debug Hooks**:  
   In debug mode, it maintains linked lists of who has which lock, pass values, line numbers, and can check for potential deadlocks across all InnoDB locks (including mutexes and rw-locks).

---

## Key Takeaways

1. **`lock_word` Encodes All Lock States**  
   This is an unusual but powerful design: one integer representing everything from “no one locked” to “N readers locked” to “exclusive locked” or “waiting writer.”  

2. **Atomic Operations + Spin + OS Wait**  
   The lock tries to keep contention overhead low by spinning a bit before resorting to an OS-based sleep/wake.  

3. **Integration with Global Wait Array**  
   When a lock can’t be acquired after some spinning, the thread goes into the `sync0arr` wait array, which allows the InnoDB server to monitor all waits.  

4. **Debugging**  
   If compiled with debug flags, there are additional structures (`rw_lock_debug_t`) that track exactly who holds the lock, who is waiting, etc. This helps detect issues like deadlocks or too-long lock waits.

Hence, `sync0rw.c` is not just an “array” of read-write locks. It’s an **implementation of a sophisticated read-write lock** with custom concurrency logic, atomic updates, spin loops, embedded OS events, debugging features, and ties to the global InnoDB wait array.

# sync0sync.c

Below is an explanation of **sync0sync.c**, which implements **InnoDB’s custom spin-based mutex** along with various synchronization primitives. We’ll look at how it differs from a standard “mutex” and see how InnoDB manages instrumentation, debugging, latch-order checks, and so on.

---

## Overview: Why a Custom Spin-Lock Mutex?

1. **Performance**:  
   - Calling OS-level semaphores or pthread mutexes can be relatively costly (microseconds per lock/unlock).  
   - A **spin lock** can be faster if the lock is expected to be held only briefly, especially on multi-CPU systems (busy-waiting may be cheaper than a thread context switch).

2. **Hybrid (Spin + OS Wait)**:  
   - InnoDB’s spin-lock code spins for a short time (a few microseconds worth of CPU cycles).  
   - If it still can’t acquire the mutex, it then **falls back** to an OS-level wait, using InnoDB’s global wait array (`sync_primary_wait_array`).

3. **Instrumentation & Debugging**:  
   - InnoDB keeps counters for spin rounds, OS waits, etc.  
   - If compiled with `UNIV_SYNC_DEBUG` or `UNIV_DEBUG`, it also tracks which thread holds which latch, file/line info, latch ordering, and so forth.

---

## Key Concepts in `sync0sync.c`

### 1. **`mutex_t` Structure**

A typical `mutex_t` in InnoDB looks like:

```c
typedef struct mutex_struct {
# if defined(HAVE_ATOMIC_BUILTINS)
    // If we have CPU atomic instructions, we use them here
# else
    os_fast_mutex_t  os_fast_mutex;
#endif
    ib_uint32_t  lock_word;   // 0 => unlocked, 1 => locked
    ulint        waiters;     // Does anyone wait on this lock?
    os_event_t   event;       // OS event used for sleeping/waking
    // Debugging & instrumentation
    os_thread_id_t   thread_id; 
    // ... more fields ...
} mutex_t;
```

#### `lock_word`  
- The central atomic “flag” for whether the mutex is held (1) or free (0).  
- Some platforms support `HAVE_ATOMIC_BUILTINS`, so InnoDB can do an atomic test-and-set. Otherwise, it might use a fallback.

#### `waiters`  
- A small integer flag. If set to 1, it indicates at least one thread is waiting.  
- This is used to reduce spurious wake-ups and coordinate with the OS event mechanism.

#### `event`  
- When a thread must give up spinning and sleep, it reserves a spot in the global wait array and uses `os_event_wait(...)` on this event to be woken up when the mutex is released.

---

### 2. **Spinning + OS Wait**

When a thread calls `mutex_spin_wait()` (the heart of `mutex_enter()` logic), it:

1. **Spins** (busy-waits) for up to `SYNC_SPIN_ROUNDS`, checking if `lock_word` becomes 0. 
   - If it becomes 0, the thread attempts an atomic test-and-set. If that succeeds, the lock is acquired.  
2. If after spinning it still can’t acquire the lock:
   - It sets up a cell in the **global wait array** (`sync_primary_wait_array`) to track that the thread is waiting for this particular mutex.  
   - Sets `waiters = 1` in the mutex.  
   - Checks one last time if it can grab the mutex quickly (in case it just got freed).  
   - If not, calls `sync_array_wait_event()` => puts the thread to sleep on `mutex->event`.  
3. When the mutex is eventually released (`mutex_exit()`), InnoDB sees if `waiters == 1` and signals `os_event_set(mutex->event)`, waking up a waiting thread.

---

### 3. **Global Wait Array** (`sync_primary_wait_array`)

As in the other files (`sync0arr.c`), InnoDB uses a global wait array for all custom mutexes, read-write locks, etc.:

- Instead of each mutex having a dedicated OS condition variable, InnoDB keeps a global array of “wait cells” for threads.  
- A waiting thread reserves a cell and then blocks on the mutex’s OS event.  
- Another thread, upon unlocking, sets that event so the waiting thread can wake up.

See [the explanation in *sync0arr.c*](https://github.com/.../sync0arr.c) for more details.

---

### 4. **Mutex Lifecycle Functions**

1. **`mutex_create_func()`**  
   - Initializes the `mutex_t` structure (sets `lock_word = 0` → unlocked).  
   - Creates an OS event (`mutex->event`) so threads can wait if spinning fails.  
   - Adds the mutex to the `mutex_list` (for debug/tracking).

2. **`mutex_spin_wait()`**  
   - The main routine to “enter” a mutex.  
   - Spinning for a certain number of iterations (configurable by `SYNC_SPIN_ROUNDS`), then do an OS yield or go to sleep if not acquired.

3. **`mutex_exit()`** (not shown verbatim, but implied in code)  
   - Sets `lock_word = 0` atomically (meaning “unlocked”).  
   - If `waiters == 1`, calls `mutex_signal_object()`, which sets `waiters = 0` and `os_event_set(mutex->event)`.

4. **`mutex_free()`**  
   - Must be called only if the memory containing the `mutex_t` is about to be freed.  
   - Removes it from the global list, frees the OS event, ensures the lock is not held.

---

### 5. **Instrumentation & Statistics**

- **`mutex_spin_wait_count`**: how many times `mutex_spin_wait()` was called.  
- **`mutex_spin_round_count`**: total spin iterations across all calls.  
- **`mutex_os_wait_count`**: how many times threads had to do an OS wait for this mutex.  
- **`mutex_exit_count`**: how many times `mutex_exit()` was called.  

These counters give insight into whether the system is frequently spinning, how contested the locks are, and so on.

---

### 6. **Debug / Latch Order Checking (`UNIV_SYNC_DEBUG`)**

In debug builds, InnoDB checks **latching order** to avoid deadlocks or inconsistent locking. This is done via:

1. **`sync_thread_level_arrays`**:  
   - A global array (size = `OS_THREAD_MAX_N`) of per-thread slot. Each slot has a fixed array of latches (`SYNC_THREAD_N_LEVELS`, typically 10,000) the thread currently holds.  
   - Each latch also has a “level” (e.g., `SYNC_FSP`, `SYNC_PAGE`, etc.).  

2. **`sync_thread_add_level(...)`**:  
   - When a thread acquires a latch, it calls this function. It checks if the new latch’s “level” is consistent with the previously held latches (i.e., you can’t acquire a latch at a “lower level” if you already hold a higher-level latch in debug mode).  
   - Helps catch potential ordering violations early.

3. **`sync_thread_reset_level(...)`**:  
   - When the latch is released, it removes that latch from the thread’s array.

4. **`mutex_set_debug_info()`**, etc.:  
   - Tracks file/line info and the owning thread, printing out debugging messages if something goes wrong.

Hence, in debug builds, if you try to lock something “out of order,” the code can `ut_error()` right away.

---

### 7. **Initialization and Teardown**

- **`sync_init()`**:  
  - Creates the **global wait array** `sync_primary_wait_array`.  
  - Initializes the global lists for mutexes and read-write locks, plus special debug structures if `UNIV_SYNC_DEBUG` is on.
  
- **`sync_close()`**:  
  - Destroys the wait array, frees all known mutexes, and tears down the debugging structures.

---

## How `sync0sync.c` Differs from a Normal Mutex

1. **Integrated Spinning**  
   - A typical OS mutex does not spin by default (though some have “spin mutex” variants). InnoDB forcibly spins for a while, then yields or sleeps—aiming to reduce context-switch overhead.

2. **Global Wait Array**  
   - Standard pthread mutexes typically use per-mutex condition variables or wait queues. InnoDB merges all waiters (mutexes, RW locks) into a single array for historical performance reasons and for debugging/monitoring.

3. **Extensive Debug Hooks**  
   - For example, latching-order checks (`sync_thread_add_level()`) are unusual in a normal library mutex. InnoDB can detect if you attempt a latch at the “wrong” time based on an internal hierarchy.

4. **Instrumentation for Performance**  
   - Counters for spin loops, OS waits, etc., let InnoDB track and tune performance or detect locking bottlenecks.

5. **Recap**  
   - Instead of letting the OS handle all details, InnoDB effectively does “lightweight” atomic test-and-set + spin, with fallback to an event-based wait.  
   - This design historically was crucial for performance under high concurrency on multi-CPU systems.

---

## Summary

`syn0sync.c` is InnoDB’s **low-level custom spin-mutex implementation**, bridging the gap between simple OS calls and the highly specialized concurrency requirements of a database engine:

1. **Custom spin-based locking** to minimize overhead.  
2. **Fallback to OS-based sleep** using a global wait array.  
3. **Instrumentation**: spin-round counters, OS-wait counters.  
4. **Optional debugging**: latch-order checking, per-latch diagnostics.  
5. **Integration** with `sync0arr.c`, `sync0rw.c`, and the rest of InnoDB’s sync framework.  

This approach aims to give robust concurrency control with detailed diagnostics, helping InnoDB scale on multiple cores while still catching locking mistakes in debug builds.