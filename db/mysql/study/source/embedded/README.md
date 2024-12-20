# Embedded InnoDB 

The last time Innobase Oy released an Innodb version  
https://github.com/nextgres/oss-embedded-innodb

# Directory Structure

Directory Structure and Components:

- [btr](./btr.md): B-Tree index management, page operations, leaf/non-leaf nodes
- buf: Buffer pool management (caching pages in memory)
- dict: Data dictionary handling (metadata about tables, indexes)
- lock: Locking subsystem (row locks, table locks, lock queueing)
- log: Write-ahead logging (WAL), log buffers, checkpoints, and crash recovery
- trx: Transaction subsystem (transaction descriptors, rollback segments)
- srv: Server-like components (startup, background threads, recovery)
- fil: File management (tablespaces, file I/O primitives)
- fsp: Tablespace file space management (extent tracking, space allocation)
- ibuf: Insert buffer for secondary indexes
- mem, os, mach: Low-level abstractions for memory, OS primitives, machine-dependent code
- sync: Synchronization primitives (mutexes, rw-locks)
- mtr: Mini-transaction abstraction (small atomic page-level operations)
- pars, que: Query graph execution modules (used internally by InnoDB’s row operations, might be less critical if you’re focusing on core storage)
- row: Row storage and retrieval