# General

Below is a refined overview of the core MySQL modules that have their origins in MySQL 5.7 (and earlier) and continue into 8.0, this time referencing some of the key source files. While MySQL’s source code is extensive, these files serve as starting points to explore the legacy architecture and how it was carried forward.

### 1. Core Server and SQL Layer

- **Server Entry Point:**
  - `sql/mysqld.cc`: The main entry point for the MySQL server process. From here, the server initializes subsystems, sets up the environment, and begins accepting connections.
- **SQL Parser and Lexing:**
  - `sql/sql_yacc.yy`: Contains the grammar rules for SQL statements. Although heavily modified over time, this file has longstanding roots and is generated into `sql_yacc.cc`.
  - `sql/lex_token.cc` and `sql/lex.h`: Implement the lexical analyzer that breaks input into tokens used by the parser.
- **Optimizer and Execution:**
  - `sql/sql_select.cc`: A core file implementing much of the SELECT statement handling logic, including join optimization and query execution plan generation.
  - `sql/opt_range.cc`: Handles range optimization for indexes—identifying efficient ways to use indexes based on WHERE clauses.
  - `sql/sql_optimizer.cc`: Contains code related to cost-based query optimization and plan refinement.
  - `sql/sql_executor.cc`: Deals with execution of the prepared query plan, orchestrating reading rows from handlers and applying operations like filtering, sorting, grouping.
- **SQL Class and Table Handling:**
  - `sql/sql_class.cc`: Defines key server and session-related structures (`THD`, `Query_arena`) that are central to how statements are processed.
  - `sql/sql_base.cc`: Manages table opening, table cache logic, and schema-level manipulations that were present in earlier versions as well.
  - `sql/handler.cc`: Implements the handler interface, which provides a uniform API for the SQL layer to interact with any storage engine.

### 2. Storage Engine Interface and Engines

- **Handler Interface and Engine Abstraction:**
  - `sql/handler.h` and `sql/handler.cc`: Define and implement the abstract `handler` class, the primary interface between the SQL layer and storage engines.
- **InnoDB (Core transactional engine, pre-dating 8.0):**
  While InnoDB code is large and spread out under `storage/innobase/`, some key files are:
  - `storage/innobase/handler/ha_innodb.cc`: The main InnoDB handler implementation that connects InnoDB internals to the MySQL Server via the handler interface.
  - `storage/innobase/srv/srv0srv.cc`: InnoDB server-side control code, handles startup/shutdown logic and background threads.
  - `storage/innobase/buf/buf0buf.cc`: Manages the InnoDB buffer pool, a central part of its architecture since before 5.7.
  - `storage/innobase/btr/btr0cur.cc`: Deals with B-Tree cursor operations, fundamental to row access in InnoDB.
  - `storage/innobase/trx/trx0trx.cc`: Transaction management logic (start, commit, rollback), which has been core to InnoDB for many years.
- **MyISAM and Other Legacy Engines:**
  Although less prominent now, these were part of the original MySQL stack:
  - `storage/myisam/mi_open.c` and other `mi_*.c` files: Implement MyISAM table handling. Still present in MySQL 8.0 source for backward compatibility and certain system tables.
  - `storage/myisam/ha_myisam.cc`: The handler interface implementation for MyISAM.

### 3. Transaction and Concurrency Control

- Concurrency and Locking (InnoDB):

  Transaction and locking code can be found mostly under storage/innobase/trx lock directories.
  
  - `storage/innobase/lock/lock0lock.cc`: The lock manager, responsible for row-level locks and transaction concurrency handling, inherited from earlier InnoDB versions.

  Even in 8.0, these core concurrency control mechanisms stem from the legacy codebase that existed in 5.7.

### 4. Replication and Logging

- **Binary Log and Replication:**
  - `sql/log.cc`: Core binary logging routines (writing transactions to the binlog).
  - `sql/log_event.cc`: Defines the structure and handling of replication events (row-based, statement-based events), logic that predates 8.0.
  - `sql/rpl_*` files such as `sql/rpl_slave.cc` and `sql/rpl_master.cc`: Implement replication threads and event handling, building on a legacy framework that was already in place before MySQL 8.0.
- **Error and General Logging:**
  - `sql/mysqld_error.h` and related files: Contain definitions and handling for server error messages.
  - `sql/log_filter.cc`, `sql/log_builtins.cc`: Integrate filters and output formats for logs, evolving from earlier versions but conceptually the same system of logs.

### 5. Security, Privileges, and Authentication

- Privilege System:
  - `sql/sql_acl.cc`: Historically central to user privileges and authentication. It defines the data structures and logic for loading/storing grants, user privileges, and more.
  - `sql/auth/sql_authentication.cc` and `sql/auth/sql_auth_cache.cc`: Contain authentication plugin frameworks and caching logic. The plugin-based authentication system has roots pre-8.0.

### 6. Plugin Architecture

- Plugin Interface:
  - `sql/sql_plugin.cc`: Implements the plugin loading and management framework.
    Plugins, including storage engines and authentication methods, were integrated this way well before MySQL 8.0.

### 7. Information Schema and Performance Schema

- **Information Schema:**
  - `sql/sql_show.cc`: Handles SHOW statements and many Information Schema views.
  - `sql/dd/*` (Data Dictionary in 8.0) replaced many old metadata structures, but `sql_show.cc` and related logic for presenting metadata via I_S tables existed in 5.7. You can still reference older code branches to see how Information Schema was generated from FRM files in previous versions.
- **Performance Schema:**
  - `storage/perfschema/*.cc`: Implementation of Performance Schema tables and instrumentation points. Although expanded over time, the Performance Schema’s core files and instrumentation concepts were introduced pre-8.0.

### 8. Core Utility Libraries

- mysys and dbug:
  - `mysys/my_sys.h`, `mysys/my_init.c`, `mysys/my_open.c`, etc.: Core utility functions for file I/O, memory handling, string operations that the server has relied on for decades.
  - `dbug/dbug.c`: Debugging macros and facilities that originated in much older versions of MySQL and still persist.

------

**In Summary:**
The listed files provide anchor points in the codebase to understand the older architecture and modules inherited by MySQL 8.0 from previous versions. By starting with these files—`mysqld.cc` for the server, `sql_yacc.yy` and `sql_select.cc` for the query layer, `handler.cc` and `ha_innodb.cc` for the storage engine interface, `log.cc` and `log_event.cc` for replication, `sql_acl.cc` for privileges, `sql_plugin.cc` for plugins, `sql_show.cc` for Information Schema, and the `mysys/` utility files—you can trace the evolution from earlier MySQL versions through to 8.0 and gain a deep structural understanding of MySQL’s internals.


# Innodb

Below is a more detailed and structured look at InnoDB’s internals, focusing on the architecture as it existed before and continued into MySQL 8.0. While there have been many enhancements in MySQL 8.0, the foundational InnoDB components—buffer management, transactional storage, logging, locking, B-Tree structures, and so forth—remain conceptually similar to earlier versions (like 5.7). We’ll examine key directories and files that make up InnoDB’s complex internal design and point out how each part fits into the bigger picture.

**Location of InnoDB Code:**
You’ll find InnoDB’s source code primarily in `storage/innobase/` within the MySQL source tree. Within this directory, it’s subdivided into functional areas, each with a set of `.cc` and `.h` files. Many of these directories and files have remained stable entry points into InnoDB’s logic through multiple versions.

------

### High-Level Architecture

InnoDB is a transactional storage engine that implements ACID semantics and uses an MVCC (Multi-Version Concurrency Control) model. Its architecture encompasses several key subsystems:

1. **Memory Management and Buffer Pool**
2. **Data Files and Page Formats**
3. **B-Tree Indexes and Row Formats**
4. **Transaction and Locking Systems**
5. **Logging and Recovery**
6. **Background Threads, IO, and Services**
7. **Dictionary and Metadata Handling**
8. **Purge, Garbage Collection, and MVCC Maintenance**

Let’s break these down into the related code components.

------

### 1. Buffer Pool and Memory Management

The buffer pool is central to InnoDB’s performance—caching database pages in memory to reduce disk I/O.

- Key Files:
  - `storage/innobase/buf/buf0buf.cc`: Core buffer pool management logic, including page allocation, eviction, and LRU management.
  - `storage/innobase/buf/buf0lru.cc`: LRU (Least Recently Used) list operations, controlling how pages are aged out.
  - `storage/innobase/include/buf0buf.h`: Declarations of buffer pool data structures, such as `buf_pool_t`.
- **Concepts:** The buffer pool is a collection of memory blocks (pages) that mirror on-disk pages. It uses sophisticated algorithms to decide which pages to keep in memory. This subsystem existed well before 8.0 and is still structured similarly, though refined over time.

------

### 2. Data Files, Page Management, and File IO

InnoDB stores data in data files (often `ibdata` files or `.ibd` files) and manages them at the granularity of pages (commonly 16KB each).

- **Key Files:**
  - `storage/innobase/fil/fil0fil.cc`: File space management code. Deals with file segments, space allocation, and header/trailer pages.
  - `storage/innobase/fil/fil0crea.cc`: Handling creation and initialization of data files.
  - `storage/innobase/include/fil0fil.h`: Definitions of file space structures.
- **IO and OS Abstraction:**
  - `storage/innobase/os/os0file.cc`: OS-level file IO wrappers. Abstracts the OS-level I/O operations so that higher-level code doesn’t need to worry about platform differences.

These files handle how pages are read from and written to disk and how free space is managed. Much of the logic for mapping logical pages to file offsets and performing asynchronous I/O traces back to legacy code.

------

### 3. Indexes, Row Format, and Data Access (B-Tree Layer)

InnoDB primarily uses B+Trees for indexing. Data and indexes are stored together in clustered indexes (the primary key index) and secondary indexes.

- **Key Files:**
  - `storage/innobase/btr/btr0cur.cc`: Core B-Tree cursor operations—used to navigate the tree, search for records, and perform inserts/deletes.
  - `storage/innobase/btr/btr0pcur.cc`: Positioning cursors within B-Trees for scans and traversal.
  - `storage/innobase/row/row0sel.cc`: Implements row selection (fetching rows that match certain search criteria).
  - `storage/innobase/row/row0upd.cc`: Row-level updates (INSERT, UPDATE, DELETE) logic.
  - `storage/innobase/include/btr0btr.h` and `row0row.h`: Contain structure definitions for B-Tree nodes and row formats.
- **Record and Page Format:**
  - `storage/innobase/page/page0page.cc`: Functions to manipulate page headers, record directory, free space, etc.
  - `storage/innobase/page/page0cur.cc`: Navigating within a page’s record directory.

These modules shape how data is physically organized and accessed. The B-Tree logic, record format, and page management code have their roots in the earliest InnoDB implementations.

------

### 4. Transaction and Locking System (MVCC and Concurrency Control)

InnoDB’s transactional model is built on MVCC. Transactions see a consistent snapshot of the database, and locks manage concurrent access.

- **Key Files:**
  - `storage/innobase/trx/trx0trx.cc`: Core transaction handling—starting, committing, rolling back transactions.
  - `storage/innobase/trx/trx0roll.cc`: Rollback segment management. Handles undo logs (where old versions of rows are stored).
  - `storage/innobase/lock/lock0lock.cc`: Lock manager implementation. Controls row-level locks and coordinates concurrency.
  - `storage/innobase/include/trx0trx.h` and `lock0lock.h`: Data structures for transactions (`trx_t`) and locks.
- **Isolation and Undo Logs:**
  - `storage/innobase/trx/trx0undo.cc`: Manages undo records that allow rollback and versioning for MVCC.

This subsystem ensures that readers and writers don’t block each other unnecessarily while maintaining data integrity and transactional guarantees. The logic is longstanding and only incrementally improved.

------

### 5. Logging and Recovery

InnoDB’s logging system records changes in a redo log to ensure durability. In the event of a crash, InnoDB uses this log to recover to a consistent state.

- **Key Files:**
  - `storage/innobase/log/log0log.cc`: Core redo logging logic. Responsible for writing redo log records and coordinating checkpointing.
  - `storage/innobase/log/log0recv.cc`: Recovery logic—parses the redo logs after a crash and applies changes to bring data files up to date.
  - `storage/innobase/include/log0log.h`: Definitions for log-related data structures.
- **Concepts:** The redo logs ensure that committed transactions are durable. By replaying logs, InnoDB can recover from unexpected interruptions. This subsystem existed in the pre-8.0 architecture and has maintained its core design principles.

------

### 6. Background Threads, Purge, and Utilities

InnoDB uses various background threads for tasks such as flushing dirty pages, purging old row versions, and performing asynchronous I/O requests.

- **Key Files:**
  - `storage/innobase/srv/srv0srv.cc`: InnoDB main server loop. Initializes the InnoDB engine, starts background threads (IO threads, page cleaner threads, etc.).
  - `storage/innobase/srv/srv0start.cc`: Startup routines, including recovery invocation.
  - `storage/innobase/sync/sync0rw.cc`: Reader-writer lock and mutex logic used internally throughout the engine.
- **Purge Thread:**
  - `storage/innobase/trx/trx0purge.cc`: Purge thread logic, which removes old versions of rows that are no longer needed by any transaction. This is essential for MVCC cleanup.

These background mechanisms were fundamental to InnoDB’s operation long before MySQL 8.0.

------

### 7. Data Dictionary and Metadata

Historically, InnoDB maintained its own internal data dictionary tables (`SYS_*` tables). In MySQL 8.0, the dictionary was integrated into the server’s transactional data dictionary, but many underlying concepts remain.

- Key Files (Pre-8.0 style, still conceptually relevant):
  - `storage/innobase/dict/dict0dict.cc`: Dictionary cache operations, loading table/index metadata.
  - `storage/innobase/dict/dict0load.cc`: Loading dictionary entries from persistent metadata tables.

These components manage how table and index metadata is stored and retrieved.

------

### 8. Utility and Common Code

InnoDB also has utility code that provides various helper functions and data structures.

- Key Files:
  - `storage/innobase/ut/ut0mem.cc`, `ut0lst.cc`, `ut0vec.cc`: Basic utility data structures (lists, arrays, memory handling).
  - `storage/innobase/include/ut0dbg.h`: Debugging macros and methods.

These utilities support all the other layers.

------

**In Summary:**
InnoDB is a layered system, and the architecture that existed in MySQL 5.7 and earlier versions is still reflected in MySQL 8.0’s code structure. By examining files like `buf0buf.cc` (buffer pool), `btr0cur.cc` (B-Tree cursors), `trx0trx.cc` (transactions), `log0log.cc` (logging), and `srv0srv.cc` (the main engine loop), you can see how the fundamental design persists. Each directory under `storage/innobase/`—`buf`, `btr`, `dict`, `fil`, `log`, `row`, `trx`, `lock`, `srv`, `ut`—represents a major facet of InnoDB’s functionality, and the code in these directories weaves together to implement a robust, ACID-compliant storage engine. Starting from these files and reading their corresponding headers (`.h` files), you can dig into the data structures and follow the call chains to understand the internal logic of InnoDB in detail.