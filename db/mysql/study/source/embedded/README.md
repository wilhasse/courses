# Embedded InnoDB 

The last time Innobase Oy released an Innodb version  
https://github.com/nextgres/oss-embedded-innodb

# Directory Structure

| Folder  | Description                                                  |
| ------- | ------------------------------------------------------------ |
| [mach](./mach.md)    | Handles low-level, machine-dependent operations and fundamental constants. Abstracts processor architecture details, byte ordering, and basic arithmetic helpers to keep higher layers platform-agnostic. |
| [include](./include.md) | Shared header directory providing declarations, macros, and common definitions used throughout the entire InnoDB codebase. Centralizes function prototypes and data structures. |
| ut      | Utility directory containing general-purpose helper routines such as data structure implementations (lists, arrays, hash), error handling utilities, and frequently reused algorithms. |
| mem     | Manages InnoDB's specialized memory handling, including custom allocators, memory pools, and debugging mechanisms (like memory poisoning) to ensure consistent and efficient allocation patterns. |
| sync    | Handles synchronization primitives such as mutexes, read-write locks, and condition variables. Provides concurrency control mechanisms at a low level. |
| thr     | Focuses on thread-related abstractions, offering portable wrappers or helpers for threading functionality to abstract OS-specific threading APIs. |
| os      | Consolidates operating-system-dependent routines, dealing with file I/O wrappers, OS-specific error handling, and system calls for cross-platform compatibility. |
| mtr     | Mini-transaction system used internally by InnoDB to group small sets of changes together. Provides core logic for atomic page operations. |
| [page](./page.md)  | Manages InnoDB's fundamental data unit: the page. Covers page format details, headers, and functions for reading, writing, and manipulating pages in memory. |
| fil     | File-layer abstraction tracking open data files, managing file segments, and providing basic file-based I/O routines. |
| fsp     | File space management responsible for logical allocation inside tablespaces, implementing extents, segments, and free space management. |
| log     | Provides redo logging infrastructure, defining change recording in transaction logs, redo record formatting, and recovery processes. |
| data    | Contains lower-level routines for handling row formats and field types, including code for reading/writing fields and data conversions. |
| dict    | Holds data dictionary logic representing metadata about tables, indexes, and columns. Maps logical SQL objects to internal structures. |
| [btr](./btr.md)   | Implements B+Tree index structures used for storing and retrieving table and index data, providing insertion, deletion, and search operations. |
| [buf](./buf.md)     | Buffer pool management subsystem controlling page caching in memory, eviction strategies, and overall memory usage. |
| ibuf    | Insert buffer optimization structure for secondary index entries, enhancing performance for write-heavy workloads. |
| row     | Handles row-level operations including creation, updates, deletions, and versioning details. Manages physical storage of records. |
| lock    | Implements the lock manager handling row-level locking, lock queues, deadlock detection, and concurrency control logic. |
| trx     | Core transaction manager orchestrating transaction lifecycle, integrating with logging, locking, and buffer pool. |
| eval    | Provides helper routines for expression evaluation, used for evaluating conditions or internal expressions. |
| read    | Contains logic for efficient row reading, including prefetch and read-ahead mechanisms. |
| que     | Query execution subsystem handling internal query-like or procedural flows within the storage engine. |
| [rem](./rem.md)     | Contains remote or "remnant" code related to row-based operations or older replication-based features. |
| ddl     | Handles Data Definition Language operations, orchestrating changes to dictionary, files, and data structures. |
| dyn     | Contains code for dynamic memory structures and string handling, including expanding buffers and specialized memory routines. |
| pars    | Responsible for parsing InnoDB-specific syntax or structures, especially internal SQL-like commands. |
| api     | External/top-level API layer providing entry points for higher-level components to interact with storage engine features. |
| srv     | Contains server control routines including background thread orchestration and initialization/shutdown sequences. |
| ha      | Houses the MySQL "handler" interface bridging MySQL Server code and InnoDB's internal APIs. |
| fut     | Collects experimental or "future" code that might not be fully integrated, including prototypes and new features. |
| usr     | Includes user-facing routines and interfaces, such as plugin APIs and debugging tools. |
