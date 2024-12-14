

# Description

Below is an in-depth, technical explanation focusing on selected parts of `handle.cc`. This file is part of MySQL’s internal server code responsible for coordinating with storage engines (like InnoDB, MyISAM, etc.) through a uniform interface represented by `handler` objects. The code deals with tasks such as reading/writing rows, handling indexes, performing multi-range reads (MRR), supporting handler-specific operations, and bridging between the SQL layer and storage engines.

Keep in mind that `handle.cc` is very large and covers a broad range of functionalities. We’ll highlight key aspects and provide examples to illustrate their usage. After this explanation, we’ll suggest next steps for deeper analysis of the source code.

### Overview of Handler Objects

**What it does:**
`handler` is an abstract class representing a storage engine instance for a given opened table. Each table opened by MySQL has a `handler` object that encapsulates how queries read, write, update, and delete rows. Different storage engines (like InnoDB) provide their own `handler` implementations.

**Key Points:**

- A `handler` object is created when a table is opened. It is associated with a `TABLE` and a `TABLE_SHARE` (metadata about the table’s structure).
- The `handler` class defines virtual methods like `write_row()`, `update_row()`, `delete_row()`, `index_read()`, `rnd_next()`, etc. Storage engines implement these methods.
- By calling these methods through the `handler` interface, the SQL layer is decoupled from the specifics of each storage engine.

### File Extensions and Discovery Mechanism

**What it does:** The code tries to understand what file extensions a storage engine uses, how to discover tables that exist in an engine’s storage but not yet known to MySQL, and how to find files to display or drop.

**How it works (example):**

- The `ha_discover()` function attempts to discover a table by calling each engine’s `discover()` callback if available.
- If you call `ha_create_table_from_engine()`, it uses `ha_discover()` to see if the engine can provide the table definition and “import” it into MySQL.
- This supports scenarios where a table exists physically in storage but is not yet registered in the data dictionary. The discovery process tries to create a .frm (pre-MySQL 8.x format) or a metadata object.

**Technical Detail:** `plugin_foreach()` loops over all registered storage engine plugins. For each engine, it checks if the engine supports `discover()` or `find_files()` and calls these methods. If any returns success, MySQL imports those table definitions into its metadata.

### Multi-Range Read (MRR) and Disk-Sweep MRR

**What it does:** MRR is an optimization to read multiple ranges of keys from a table more efficiently, reducing random I/O. The code in `handle.cc` deals with setting up and executing MRR. The `DsMrr_impl` class (Disk-Sweep MRR) is a specialized MRR implementation that sorts rowids and attempts to read them in a more sequential manner.

**How it works (example):**

- Suppose you have a query with multiple key ranges: `SELECT * FROM t WHERE key_col BETWEEN 10 AND 20 OR key_col BETWEEN 30 AND 40`. MySQL’s MRR attempts to gather these ranges, execute them more efficiently, and reduce random reads.
- The `ha_multi_range_read_init()` method initializes MRR. The `ha_multi_range_read_next()` fetches the next row from the selected ranges.
- `DsMrr_impl` sorts references to rows, tries to read them in disk order, and reduces costly random I/O.

**Technical Detail:** `DsMrr_impl::dsmrr_fill_buffer()` retrieves rowids for ranges and sorts them. `choose_mrr_impl()` decides if the disk-sweep strategy is cost-effective. If so, MySQL uses it instead of the default MRR. This is a cost-based decision considering system variables and optimizer switches.

### Transactional Support, Locking, and Binlog Integration

**What it does:** When you run `INSERT`, `UPDATE`, or `DELETE`, the handler code calls methods like `ha_write_row()` or `ha_update_row()`. These wrap storage engine calls and also handle binary logging, transaction states, and lock modes.

**How it works (example):**

- `ha_write_row()` calls the engine’s `write_row()` method to insert a row. On success, it then calls `binlog_log_row()` if binary logging is enabled and row-based logging is being used.
- `ha_external_lock()` coordinates with the engine to apply external locks for concurrency control at the engine level. This ensures consistent reads and correct isolation levels.

**Technical Detail:**

- The code checks if the current statement should be logged in the binary log and calls the right `log_func` (like `Write_rows_log_event`) for row-based replication.
- Before an engine’s transaction is committed, MySQL might call `ha_commit_trans()` which calls `prepare()`, `commit()` or `rollback()` for each involved handler.

### Secondary Engines and Handlerton Structures

**What it does:** `handlerton` is a structure representing a storage engine’s capabilities and callbacks. `handle.cc` frequently uses `handlerton` functions to determine what each engine supports (e.g., `notify_exclusive_mdl()`, `is_reserved_db_name()`) and to handle special operations per engine.

**How it works (example):**

- When MySQL locks a table or performs ALTER TABLE, it might notify the storage engines via `ha_notify_table_ddl()` or `ha_notify_exclusive_mdl()` so engines can take internal actions.
- For system tables or foreign key constraints, `ha_check_if_supported_system_table()` determines if an engine can handle specific internal system tables.

**Technical Detail:**

- `handlerton` pointers point to engine-specific structs. By calling `plugin_foreach(...)` MySQL loops through all engines and tries to execute engine-specific notifications or capabilities checks.

### General Utility and Status Reporting

**What it does:** `handle.cc` also has utility code for reporting engine status via `ha_show_status()`, error handling, adjusting auto-increment values (`update_auto_increment()`), and other housekeeping tasks like temporary file removal (`ha_rm_tmp_tables()`).

**How it works (example):**

- `ha_show_status()` calls `show_status()` on all engines to gather engine-level statistics.
- `ha_check()` and related methods check for table consistency and initiate repairs if needed.

**Technical Detail:** When an error occurs (e.g., `HA_ERR_KEY_NOT_FOUND`), `handler::print_error()` maps internal error codes to user-facing SQL error messages. This ensures uniform error reporting regardless of which engine returned the error.

### Example Workflow

1. **Table Open:** A SELECT query opens a table. MySQL allocates a `handler` for the table and calls `ha_index_init()` if an index scan is needed.
2. **Read Rows:** `ha_index_read()` reads the first row of a range. `ha_index_next()` reads subsequent rows, checking end conditions.
3. **Multi-Range Read:** If multiple key ranges are requested, `multi_range_read_init()` sets up MRR and might use disk-sweep MRR if cost-effective.
4. **Modifying Data:** For an INSERT, `ha_write_row()` calls `write_row()` in the engine, then logs it to the binlog if needed.
5. **Transactions:** At commit time, `ha_commit_trans()` calls each involved engine’s commit function. If replication or binary logging is enabled, it ensures all changes are properly logged and synced.

### Next Steps for Analysis

1. **Read `handler.h`:**
   Examine `handler.h` to understand the class definition of `handler` and its virtual methods. This gives you the blueprint that `handle.cc` implements.
2. **Study Engine Plugins:** Look at `innodb_handler.cc` or `myisam_handler.cc` to see how storage engines implement `handler` methods. This helps you understand how `handle.cc` ties into real engine operations.
3. **Transaction Coordinator (`tc_log`):** Explore files like `tc_log.cc` to see how distributed transactions and 2-phase commit logic integrate with handler calls in `handle.cc`.
4. **Optimizer and Cost Model:** Investigate the cost model interfaces (e.g., `sql/opt_costmodel.h`) to learn how the MRR cost decisions are made and how this affects `handle.cc` logic.
5. **Replication and Logging Code:** Look into `binlog.cc` and `log_event.cc` to see how row events are formed and how `ha_binlog_log_query()`, `binlog_log_row()` integrate with replication.
6. **System Table Handlers:** Check `sql/dd` directory to see how data dictionary interactions are performed. Understanding `ha_discover()` also means understanding MySQL’s metadata dictionary interactions.

By following these steps, you can gain a comprehensive understanding of how `handle.cc` plays a central role as the liaison between SQL commands, the optimizer, the transaction manager, and the storage engines.


# Technical


Below is an even more technically focused explanation of `handle.cc`, diving deeper into internal mechanisms, data structures, and control flows.

### Deeper Technical Details

#### Handler Class Architecture

**`handler` class and `handlerton`:**

- Each storage engine is represented by a `handlerton` structure, which is a global descriptor that includes function pointers for engine-specific operations (e.g., `discover`, `start_consistent_snapshot`, `drop_database`, etc.).
- When a table is opened, MySQL uses the table’s `TABLE_SHARE` and the storage engine’s `handlerton` to create a `handler` instance. This `handler` object serves as a per-table interface into the engine.

**Data Structures:**

- `TABLE` and `TABLE_SHARE`:
  - `TABLE_SHARE`: Holds metadata about the table structure (columns, keys, etc.) and is shared among all references to the same table.
  - `TABLE`: Represents an open table instance. It includes pointers to `handler`, `read_set` and `write_set` bitmaps (indicating which columns are read/written), row buffers (`record[0]`, `record[1]`, etc.), and per-query context.
- `handler`:
  - An instance of a derived class (e.g., `ha_innodb` for InnoDB) that implements virtual functions like `write_row()`, `update_row()`, `delete_row()`, and indexing methods (`index_read()`, `index_next()`).
  - `handler::ref` and `handler::dup_ref`: Buffers to store position references (like RIDs or primary key values) for current rows in index scans.

#### Locking and External Locking

**`ha_external_lock()`:**

- Called when the SQL layer obtains (or releases) a table lock (e.g., `LOCK TABLES t WRITE`).
- The engine may perform engine-level locking or ensure transactional consistency.
- After obtaining an external lock with `F_WRLCK` or `F_RDLCK`, subsequent reads and writes by the `handler` must ensure isolation and safety.

**Concurrency and Isolation Levels:**

- The handler must cooperate with MySQL’s chosen isolation levels (e.g., `REPEATABLE-READ`).
- Some engines support gap locks or next-key locks, while others do not. `handle.cc` checks if a certain pattern of lookup is allowed without gap locks. For example, if the engine does not support gap locks but a repeatable-read transaction tries to do a partial key lookup that could require them, MySQL may raise an error or enforce different conditions.

#### Index Operations

**Index Scan Methods:**

- ```
  ha_index_read_map()
  ```
  ```
  ha_index_next()
  ```
  ```
  ha_index_prev()
  ```
  ```
  ha_index_first()
  ```
  ```
  ha_index_last()
  ```

  - These methods start or continue an index scan.
  - `ha_index_read_map()` finds the initial position in the index based on a key value.
  - `ha_index_next()` fetches the next row in index order.
  - During these calls, `MYSQL_TABLE_IO_WAIT()` macros register performance schema instrumentation events.

**Key Comparison:**

- ```
  compare_key()
  ``` 
  ```
  key_cmp()
  ```
  :

  - After retrieving a row, MySQL may need to check if it is still within the desired key range.
  - `compare_key()` compares the current index position to the end range. If the position has passed the upper bound, it returns `HA_ERR_END_OF_FILE`.

**Partial Keys and NULL handling:**

- The code carefully handles partial key specifications (keypart maps) and the presence of NULL values. Certain queries might use partial indexes (like `WHERE key_col = const`) or `>=`, `<=` conditions, requiring different `find_flag` operations (`HA_READ_KEY_EXACT`, `HA_READ_AFTER_KEY`, etc.).

#### Multi-Range Read (MRR)

**`multi_range_read_info()` and `multi_range_read_init()`:**

- The MRR interface is a cost-based optimization. MySQL can read multiple key ranges efficiently, batching them together.
- Two main implementations: The default "Range MRR" and the "Disk-sweep MRR" (`DsMrr_impl`). Disk-sweep MRR tries to reorder row retrieval to reduce random I/O.

**`DsMrr_impl`:**

- Uses a temporary buffer to store rowids for all keys in the given set of ranges, sorts them, and attempts to read rows in sorted order.
- This can drastically reduce random page reads at the cost of memory and CPU for sorting.
- The `choose_mrr_impl()` function decides whether to use the default MRR or Disk-sweep MRR based on cost estimates and optimizer switches.

**MRR Call Flow Example:**

1. `ha_multi_range_read_init()` sets up the MRR and possibly clones the handler (`handler::clone()`) to use a separate index cursor for initial range lookups.
2. `ha_multi_range_read_next()` repeatedly fetches the next row from the batched ranges, possibly refilling buffers and sorting rowids when needed.

#### Transaction Handling and Binlogging

**Transaction Hooks:**

- ```
  ha_commit_trans()
  ```
  ```
  ha_rollback_trans()
  ```

  - At commit or rollback, MySQL loops through all involved handlers and calls their `commit()` or `rollback()` methods.
  - `tc_log->prepare()` and `tc_log->commit()` (TC stands for Transaction Coordinator) handle 2-phase commit if multiple engines are involved.

**Row-based Replication (RBR):**

- Before logging row changes (`INSERT`, `UPDATE`, `DELETE`) to the binary log, `ha_write_row()`, `ha_update_row()`, and `ha_delete_row()` call `binlog_log_row()`.
- `binlog_log_row()` writes a table map event if needed and then logs the row event. The handler ensures that binlog events are only written if the operation affects a table that is replicated and the binlog is enabled.

**Conflict Handling and Duplicate Keys:**

- If a handler returns `HA_ERR_FOUND_DUPP_KEY`, `handler::print_error()` translates this error into a meaningful SQL error like `ER_DUP_ENTRY`.
- In special cases (e.g., foreign key constraints), the handler might need to produce more specific error messages or run additional checks.

#### Memory Management and Temporary Buffers

**Handling Memory and Workspaces:**

- The code often allocates temporary memory for row buffers, arrays, and sorting buffers using `mem_root` allocators tied to the table’s lifetime.
- `handler::extra()` and related calls can signal to the engine that bulk operations are starting (`HA_EXTRA_BEGIN_ALTER`), or that buffering behaviors can be changed (`HA_EXTRA_NO_KEYREAD`, etc.).

**Auto-increment Handling:**

- `update_auto_increment()` calculates and reserves auto-increment values based on concurrency and requested increments.
- If the handler supports atomic increments or uses a shared counter, MySQL updates these values accordingly and ensures binlog consistency.

#### Error Handling and Diagnostics

- `handler::print_error()` maps handler error codes (e.g., `HA_ERR_KEY_NOT_FOUND`) to SQL layer error messages.
- Some errors like `HA_ERR_LOCK_DEADLOCK` trigger retries or produce warnings instead of immediate query failures, depending on the context.

#### System Table and Dictionary Integration

**`ha_discover()`, `ha_is_externally_disabled()`:**

- These functions allow MySQL’s dictionary to discover tables known only by the engine (for legacy reasons or engine-specific metadata).
- If an engine is externally disabled or the user tries to access system tables that the engine doesn’t support, MySQL issues errors or falls back to a default engine.

**In-Place Alterations and Notifications:**

- `prepare_inplace_alter_table()` and `commit_inplace_alter_table()` notify engines before and after online DDL operations.
- `ha_notify_table_ddl()` calls `notify_alter_table()` or `notify_rename_table()` so engines can do internal actions (like updating internal dictionaries or rebuilding indexes).

### Next Steps for In-depth Analysis

1. **Tracing a Full Query Execution Path:**
   - Start from the parser (`sql_parse.cc`) that constructs a query plan, see how it chooses indexes and calls `join_read_*` functions that eventually call `ha_index_read()` and others.
   - Understand how `handler::ha_*` methods integrate into the optimizer and executor steps (like `filesort()`, `ref_optimizer()`, and `range_optimizer()`).
2. **Engine-specific Code Comparison:**
   - Compare this generic code in `handle.cc` with `innobase/handler/ha_innodb.cc` or `myisam/ha_myisam.cc` to see how engines implement these methods differently.
3. **Testing with Different Engines and Configurations:**
   - Use a MySQL debug build and run tests with `--debug=d,...` to trace calls into `handle.cc`.
   - Experiment with optimizer switches (`optimizer_switch='mrr=on/mrr=off'`) and inspect performance.
4. **Performance Schema and Instrumentation:**
   - Look at the `MYSQL_TABLE_IO_WAIT` macros. They tie into Performance Schema (PSI instrumentation). Checking `mysql/psi/mysql_table.h` shows how these are recorded and can be analyzed using Performance Schema tables.

By examining these technical details, one gains insight into how `handler` mediates between the SQL execution layer and storage engines, ensuring consistent locking, transactional semantics, indexing operations, and integration with replication and the optimizer. This understanding is crucial for diagnosing performance issues, implementing custom storage engines, or enhancing MySQL’s internal operations.