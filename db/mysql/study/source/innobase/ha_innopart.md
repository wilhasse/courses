## Header



In MySQL, tables can be partitioned so that the data is split into multiple underlying physical tables (partitions), each typically residing in its own separate file (if file-per-table is enabled). InnoDB, as a storage engine, supports native partitioning. The `ha_innopart` class defined in this header is the "handler" class that integrates InnoDB’s storage capabilities with MySQL’s partitioning layer.

MySQL handlers are interfaces that connect MySQL’s server layer operations (INSERT, SELECT, UPDATE, etc.) to the underlying storage engine. For a partitioned table, there must be logic to determine which partition to operate on, handle index lookups across partitions, and manage DDL (Data Definition Language) operations affecting multiple partitions.

------

### Major Components

1. **`Ha_innopart_share` Class:**

   - Inherits from `Partition_share`.
   - Represents shared metadata and structures common to all handler instances for a given partitioned table (similar to `INNOBASE_SHARE` in `ha_innodb.h`).
   - Contains arrays of `dict_table_t*` for each partition (one `dict_table_t` per partition), and arrays for index mappings.
   - Provides functions to open/close all partitions and increment/decrement reference counts.
   - Handles setup of virtual columns and index translation.

2. **Partition Helper Functions:**

   - There are static and inline helper functions, like `partition_get_tablespace()`, which determine the tablespace for a given partition or subpartition.
   - These utility functions aid in managing and retrieving partition-specific metadata.

3. **`ha_innopart` Class:**

   - This is the main class that implements the handler interface for InnoDB’s native partitioning.
   - Inherits from:
     - `ha_innobase`: The base InnoDB handler class for a single non-partitioned table.
     - `Partition_helper`: A helper class providing common partitioning functionality (like iterating over partitions, retrieving partition IDs, etc.).
     - `Partition_handler`: Interface providing partition-specific methods (e.g., `truncate_partition`, `exchange_partition`).
   - The `ha_innopart` class merges these functionalities so that all SQL operations can be performed seamlessly over multiple partitions.

   Key responsibilities include:

   - **Opening/Closing:** Overriding `open()` and `close()` to handle opening all underlying partitioned tables from the InnoDB dictionary.
   - **Index Operations:** Implementing `index_read`, `index_next`, `index_first`, etc., in a partition-aware manner. It can switch between partitions or merge results from multiple partitions.
   - **Row Operations:** `write_row()`, `delete_row()`, `update_row()` are overridden to route operations to the correct partition based on partitioning keys.
   - **Statistics and Meta-Info:** `info()` and related methods are extended to return combined statistics from all partitions.
   - **Auto-Increment Handling:** Ensures that the auto-increment counter for a partitioned table is consistent across all partitions.
   - **DDL Operations:** Deals with `TRUNCATE`, `RENAME`, `ALTER TABLE`, and `EXCHANGE PARTITION` by applying these operations to each underlying partition's InnoDB table. Special logic is needed to handle online/in-place alters.
   - **Sampling and Parallel Operations:** Includes methods like `sample_init`, `sample_next`, and `sample_end` for table sampling, as well as hooks for parallel reading.

4. **Additional Class Elements:**

   - **`saved_prebuilt_t` structure:** Keeps track of prebuilt structures for each partition. InnoDB uses a `row_prebuilt_t` structure to manage cursor positions and states; for a partitioned table, each partition may need its own state.
   - Arrays and Maps:
     - `m_parts`: An array of `saved_prebuilt_t` to store per-partition state.
     - `m_pcur_parts`, `m_clust_pcur_parts`: Arrays of persistent cursors per partition.
     - `m_pcur_map`: A mapping from partition ID to the appropriate cursor arrays.
   - Flags and Settings:
     - `m_reuse_mysql_template`: Determines if the MySQL column mapping templates can be reused across partitions.
     - `m_new_partitions`: Used to track newly added partitions during DDL operations like `ADD PARTITION`.

------

### Key Interfaces and Methods

- **Partitioning-Specific Methods:**
  - `truncate_partition_low(dd::Table *dd_table)`: Truncates selected partitions by dropping and recreating their underlying tablespaces.
  - `exchange_partition_low(...)`: Exchange the contents of one partition with another standalone table.
  - `prepare_inplace_alter_partition`, `inplace_alter_partition`, `commit_inplace_alter_partition`: Handle online/in-place ALTER TABLE partition operations.
- **Index and Row Access:**
  - Methods like `index_read_in_part`, `rnd_next_in_part`, `delete_row_in_part` implement partition-aware logic. They call `set_partition()` to switch the active partition and then delegate to `ha_innobase` methods for actual row/index operations.
- **Statistics and Autoincrement:**
  - Overrides `records()`, `scan_time()`, and other methods to sum or merge partition-level stats.
  - Autoincrement initialization and retrieval consider all partitions.
- **Concurrency and Locking:**
  - Inherits InnoDB’s concurrency and locking model, but with logic adjusted for multiple partitions. The code may involve locking tablespaces for each partition or updating states to reflect that multiple underlying tables are in use.

------

### Purpose and Importance

The `ha_innopart.h` file defines a crucial piece in enabling partitioned tables to work efficiently with the InnoDB storage engine. Without it, MySQL could not natively manage multiple InnoDB tables as a single logical partitioned table. By integrating with `ha_innobase` and `Partition_handler`, it provides:

- A unified SQL interface for partitioned InnoDB tables.
- Proper handling of complex DDL operations on partitioned tables.
- Efficient read/write operations distributed across partitions.
- Correct and consistent metadata management, auto-increment handling, and statistics gathering.

------

### Summary

In essence, `ha_innopart.h` introduces a specialized handler (`ha_innopart`) for partitioned InnoDB tables. It uses logic from the `ha_innobase` class for single-table operations and extends that logic to handle multiple physical tables (partitions) under a single logical entity. This header sets forth all member variables, internal structures, and virtual methods necessary for implementing a fully functional partitioned InnoDB storage handler in MySQL.



## Code

MySQL supports table partitioning, allowing a single logical table to be split into multiple underlying physical sub-tables called partitions. While MySQL’s generic partition handler (in `ha_partition.cc`) can manage partitioning for most engines, InnoDB has a specialized approach to handle native partitioning more directly and efficiently. The `ha_innopart.cc` file implements the `ha_innopart` class, which is a subclass of `ha_innobase` specifically tailored for native partitioned tables in InnoDB.

Native partitioning means that each partition of an InnoDB table is actually its own InnoDB table with its own `.ibd` file (if using file-per-table tablespaces), own indexes, and its own statistics. The `ha_innopart` class coordinates these multiple underlying InnoDB tables so that MySQL sees them as a single logical table.

------

### Key Components and Responsibilities

1. **`ha_innopart` Class:**
   - Inherits from `ha_innobase` and `Partition_helper`.
   - Manages multiple `dict_table_t` objects, one for each partition, and keeps track of their indexes, statistics, and autoincrement values.
   - Overrides methods like `open()`, `close()`, `write_row()`, `read_row()`, `index_read()`, `rnd_init()` etc., to handle operations across multiple partitions.
   - Integrates partition pruning logic. When the MySQL optimizer chooses which partitions to read from, `ha_innopart` can limit read operations to only those partitions, improving performance.
2. **Partition-Aware Operations:**
   - **Open/Close:** On `open()`, `ha_innopart` locates and opens all physical InnoDB partition tables. On `close()`, it releases them.
   - **Row Operations:** Operations such as `write_row()`, `delete_row()`, and `update_row()` must first determine which partition a particular row belongs to. Then they delegate the actual operation to that partition’s InnoDB table.
   - **Index Operations:** `ha_innopart` uses mapping structures to translate MySQL’s key (index) definitions to the underlying InnoDB indexes within each partition. It supports index scans, range reads, and sort operations across multiple partitions.
3. **Reference and Positioning:**
   - In non-partitioned InnoDB, a `ref` (row reference) typically corresponds to the primary key or the InnoDB internal row ID. With partitioning, the position must also encode which partition the row resides in.
   - `ha_innopart` modifies the reference storage format to include partition information. This is essential for `rnd_pos()` and `position()` calls.
4. **Statistics and ANALYZE Support:**
   - The code updates and merges statistics (like row counts and index cardinalities) from all partitions to present a unified view to the MySQL optimizer.
   - `ha_innopart` calls InnoDB’s statistics routines on each partition and merges results. This helps the optimizer make better decisions about query plans.
5. **Truncation and DDL Operations:**
   - Operations like `TRUNCATE TABLE` must be applied to all underlying partitions. This code invokes DDL-related code for each partition’s tablespace.
   - Similar logic applies to `RENAME TABLE`, `DROP TABLE`, `IMPORT`/`DISCARD TABLESPACE`, and `ALTER TABLE`. Each partition is handled individually.
6. **Full-Table Scans and Range Queries:**
   - For operations like `records_in_range()`, `scan_time()`, and full table scans (`rnd_next()`), `ha_innopart` iterates over selected partitions. If an ordered scan is required, it may internally use a priority queue to merge sorted results from multiple partitions.
7. **Error Handling and Partition Boundaries:**
   - The code includes checks to detect rows that may have been placed in the wrong partition (e.g., due to corruption or user error) and handle them during `CHECK TABLE` or `REPAIR TABLE`.
   - It integrates with MySQL’s error handling framework to return correct error codes when a partition is missing, discarded, or corrupted.
8. **Auto-increment Handling:**
   - Each partition can contain an auto-increment counter. `ha_innopart` coordinates these counters to ensure correct global sequencing. It merges the maximum AUTOINC values from all partitions and updates them as needed.
9. **Integration with Data Dictionary and Metadata:**
   - The code reads and updates metadata in MySQL’s Data Dictionary (DD). For each partition, it may perform updates to DD objects and properties, ensuring consistency between InnoDB’s internal dictionary and MySQL’s global dictionary.

------

### The Workflow in Practice

When MySQL needs to process a SELECT, INSERT, UPDATE, or DELETE on a partitioned InnoDB table:

1. The server calls `ha_innopart::open()`, which opens each partition’s InnoDB table.
2. For queries that read data, `ha_innopart` prunes partitions so that only required partitions are accessed.
3. For each selected partition, `ha_innopart` uses the standard InnoDB mechanisms (from `ha_innobase`) to access data and indexes.
4. If multiple partitions are read simultaneously, `ha_innopart` merges results. For ordered retrieval (like a sort-merge), it manages multiple per-partition cursors and a priority queue.
5. On writes, `ha_innopart` determines the target partition using the partitioning function and then invokes InnoDB write operations on that partition’s `dict_table_t`.
6. On `close()`, it releases all partition-level resources.

------

### Importance of `ha_innopart.cc`

- **Performance:** Direct handling of partitions inside the InnoDB handler can improve performance and efficiency over a generic handler approach.
- **Maintainability:** Keeping partition logic separate helps isolate the complexity of dealing with multiple underlying files and dictionary entries.
- **Correctness:** Ensures that each row operation is correctly routed to the appropriate partition. This file centralizes logic for dealing with partition boundaries and state changes.

------

### Summary

`ha_innopart.cc` is all about integrating InnoDB’s native partitioning with MySQL’s handler interface. It handles the complexity of multiple underlying InnoDB tables representing a single logical partitioned table. This includes opening/closing partitions, merging statistics, managing auto-increments, performing row and index operations on the correct partitions, handling DDL operations like truncate or rename across all partitions, and ensuring consistency with MySQL’s Data Dictionary.