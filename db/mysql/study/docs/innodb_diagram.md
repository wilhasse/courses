# Innodb Architecture

```mermaid
%% =================================
    %% Subgraph: InnoDB Transaction & Concurrency
    %% =================================
    subgraph A[Transaction & Concurrency]
        direction TB
        A1[Transaction Manager]
        A2[Lock Manager]
        A3[MVCC Multi-Version Concurrency Control]
    end

    %% =================================
    %% Subgraph: Buffer Management
    %% =================================
    subgraph B[Buffer Management]
        direction TB
        B1[Buffer Pool]
        B2[LRU & Flush Lists]
        B3[Adaptive Hash Index]
        B4[Change Buffer]
    end

    %% =================================
    %% Subgraph: Logging & Recovery
    %% =================================
    subgraph C[Logging & Recovery]
        direction TB
        C1[Redo Logs]
        C2[Undo Logs]
        C3[Doublewrite Buffer]
        C4[Checkpointing]
    end

    %% =================================
    %% Subgraph: Data Organization
    %% =================================
    subgraph D[Data Organization]
        direction TB
        D1[Data Dictionary]
        D2[Tablespaces & Data Files]
        D3[Index Structures]
    end

    %% =================================
    %% Subgraph: Background Threads
    %% =================================
    subgraph E[Background Threads]
        direction TB
        E1[Master Thread]
        E2[IO Threads]
        E3[Page Cleaner Threads]
        E4[Purge Threads]
    end

    %% =================================
    %% Links / Relationships
    %% =================================

    %% Transaction & Concurrency <--> Buffer Management
    A --> B

    %% Buffer Management <--> Logging & Recovery
    B --> C

    %% Buffer Management <--> Data Organization
    B --> D

    %% Logging & Recovery <--> Data Organization
    C --> D

    %% All feed into Background Threads for processing
    A --> E
    B --> E
    C --> E
    D --> E
```

# Explanation of Key InnoDB Components

## 1. Transaction & Concurrency
- **Transaction Manager**  
  Manages the lifecycle of transactions, ensuring ACID properties (Atomicity, Consistency, Isolation, Durability). It coordinates commits, rollbacks, and interactions with the other subsystems.

- **Lock Manager**  
  Controls row-level locking (and sometimes table-level locks) to prevent conflicting operations. Uses sophisticated algorithms (like lock granularity on record ranges) to maximize concurrency.

- **MVCC (Multi-Version Concurrency Control)**  
  Maintains multiple versions of rows to allow consistent reads without blocking writers. This mechanism uses `undo` data to reconstruct previous versions of rows for snapshot reads.

## 2. Buffer Management
- **Buffer Pool**  
  A large shared memory area used to cache pages (data and index pages) from tables on disk, significantly reducing disk I/O.  
- **LRU & Flush Lists**  
  A pair of linked lists that track which pages need to be flushed to disk (modified pages) and which pages are oldest/least recently used.  
- **Adaptive Hash Index**  
  Dynamically built in-memory hash index for frequently accessed pages to speed up lookups.  
- **Change Buffer**  
  Stores changes to secondary indexes in a buffer for later merge, reducing random disk I/O.

## 3. Logging & Recovery
- **Redo Logs**  
  Write-ahead logs (often in `ib_logfile` or combined into `ib_redo`) for crash recovery. All modifications are recorded here before the actual data pages are updated.  
- **Undo Logs**  
  Track the old versions of rows to support rollbacks and MVCC-based reads. Stored within `undo` tablespaces or inside the main tablespace, depending on configuration.  
- **Doublewrite Buffer**  
  A data integrity feature that writes pages twice to help recover from partial page writes or OS/hardware-level issues.  
- **Checkpointing**  
  Periodically flushes dirty pages from the buffer pool to disk, ensuring that the redo log does not grow indefinitely and reducing recovery time after a crash.

## 4. Data Organization
- **Data Dictionary**  
  Contains metadata about databases, tables, columns, indexes, foreign keys, etc. In MySQL 8, some of this is integrated into MySQL’s global data dictionary, but InnoDB still manages its own for internal operations.  
- **Tablespaces & Data Files**  
  Physical storage of table and index data. May be in a single shared tablespace (`ibdataX`) or multiple file-per-table tablespaces.  
- **Index Structures**  
  InnoDB primarily uses clustered indexes (the primary key index contains the row data) and secondary indexes (non-clustered).

## 5. Background Threads
- **Master Thread**  
  Oversees most major activities in InnoDB: flushing, checkpointing, purging, etc.  
- **IO Threads**  
  Handle asynchronous read/write requests (e.g., prefetching, background writes).  
- **Page Cleaner Threads**  
  Specifically focus on flushing dirty pages from the buffer pool to disk without blocking user transactions.  
- **Purge Threads**  
  Clean up old row versions and `undo` entries that are no longer needed by running transactions, helping manage the size of undo logs.

