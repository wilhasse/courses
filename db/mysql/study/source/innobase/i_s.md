The provided header (`i_s.h`) and source (`i_s.cc`) files are part of the InnoDB storage engine codebase for MySQL, focusing on the implementation of various INFORMATION_SCHEMA tables related to InnoDB. These tables expose internal InnoDB states, statistics, and metadata to database administrators through SQL queries.

Let’s break down and explain the key aspects of the code and its purpose, step-by-step and in great detail:

### Overview

InnoDB is the default storage engine for MySQL. It stores table and index metadata internally. To make these metadata and runtime statistics accessible through SQL, InnoDB implements special INFORMATION_SCHEMA tables. These tables are not backed by on-disk data; instead, they are populated dynamically at query time by C++ functions within the InnoDB codebase. The files `i_s.h` and `i_s.cc` define and implement these dynamic INFORMATION_SCHEMA tables.

Each INFORMATION_SCHEMA table is essentially a “plugin” that, when queried, calls a function that gathers the relevant in-memory data structures from InnoDB and formats them as table rows.

### The Header File (`i_s.h`)

In the header file, you see forward declarations and references to various plugin structures, such as:

```
c++Copy codeextern struct st_mysql_plugin i_s_innodb_trx;
extern struct st_mysql_plugin i_s_innodb_cmp;
extern struct st_mysql_plugin i_s_innodb_cmp_reset;
... and so on
```

Each `i_s_innodb_*` symbol refers to a different INFORMATION_SCHEMA table plugin implemented in `i_s.cc`. The `i_s_innodb_trx` table, for example, provides information about currently running InnoDB transactions. Similarly, `i_s_innodb_cmp` provides statistics about compression operations, `i_s_innodb_metrics` exposes internal performance counters, and `i_s_innodb_buffer_page` or `i_s_innodb_buffer_page_lru` shows details of the pages in the InnoDB buffer pool.

The header thus lists these plugins so that MySQL’s plugin loading mechanism can know about them and the server can reference them.

### The Source File (`i_s.cc`)

The source file `i_s.cc` is extensive. It defines and registers a collection of INFORMATION_SCHEMA tables. Each table follows a pattern:

1. **Field Definitions (ST_FIELD_INFO arrays):**
   For each dynamic INFORMATION_SCHEMA table, the code defines a static array of `ST_FIELD_INFO` structures. Each entry describes a column name, type, length, and some flags. This effectively describes the schema of the dynamic table.

   For example:

   ```
   c++Copy codestatic ST_FIELD_INFO innodb_trx_fields_info[] = {
       {"trx_id", ...},
       {"trx_state", ...},
       ...
       END_OF_ST_FIELD_INFO
   };
   ```

   This sets up the columns that will appear when a user does `SELECT * FROM information_schema.innodb_trx;`.

2. **Initialization Functions:**
   Each table has an `init` function that sets `schema->fields_info` to point to the columns defined above and sets `schema->fill_table` to a function that will populate the table’s rows at query time.

   For example:

   ```
   c++Copy codestatic int innodb_trx_init(void *p) {
       ST_SCHEMA_TABLE *schema = (ST_SCHEMA_TABLE*) p;
       schema->fields_info = innodb_trx_fields_info;
       schema->fill_table = trx_i_s_common_fill_table;
       return 0;
   }
   ```

   Here, `trx_i_s_common_fill_table` is the function that actually pulls the data about running transactions and writes it into the result set.

3. **Filling the Tables:**
   The `fill_table` functions use InnoDB’s internal data structures and APIs to gather the requested information. For instance, `trx_i_s_common_fill_table` iterates over the list of active transactions maintained by InnoDB and stores their details (transaction ID, state, start time, etc.) into the table fields. It then calls `schema_table_store_record()` to produce rows visible to the SQL layer.

   This process is read-only with respect to server state; it just formats what is currently available in memory.

4. **Plugin Registration:**
   Each I_S table is registered as a MySQL plugin using a `struct st_mysql_plugin` instance. The code sets various fields such as `name`, `author`, `init`, `deinit`, `version`, and `info`. For example:

   ```
   c++Copy codestruct st_mysql_plugin i_s_innodb_trx = {
     type: MYSQL_INFORMATION_SCHEMA_PLUGIN,
     info: &i_s_info,
     name: "INNODB_TRX",
     author: plugin_author,
     ...
     init: innodb_trx_init,
     ...
     version: i_s_innodb_plugin_version
   };
   ```

   This structure is how MySQL’s plugin framework learns about and manages the plugin. When MySQL starts up or when the plugin is installed, it calls `init` to set things up, and later on, when a user queries `INFORMATION_SCHEMA.INNODB_TRX`, MySQL routes the request to the plugin’s `fill_table` function.

### Specific Tables Explained

There are many tables defined here. Let’s look at some key examples:

- **INNODB_TRX:**
  Provides information about currently running InnoDB transactions. The fields show transaction IDs, start times, isolation levels, and the query currently being executed. It’s especially useful for diagnosing lock waits and long-running transactions.
- **INNODB_CMP and INNODB_CMP_RESET:**
  These show statistics about compression operations (how many compression/decompression attempts have been made, how long they took, etc.). The “RESET” variants also reset the counters after retrieval.
- **INNODB_BUFFER_PAGE and INNODB_BUFFER_PAGE_LRU:**
  They expose detailed information about the pages currently stored in the InnoDB buffer pool. You can see which pages belong to which tables, how recently they were accessed, their modification times, and whether they are old or young pages in the LRU list. This helps DBAs understand buffer pool content and performance tuning.
- **INNODB_TABLES, INNODB_COLUMNS, INNODB_INDEXES:**
  These tables give metadata about InnoDB’s internal data dictionary. `INNODB_TABLES` lists all known InnoDB tables, their IDs, flags, and row formats. `INNODB_COLUMNS` and `INNODB_VIRTUAL` enumerate the columns and virtual columns of these tables. `INNODB_INDEXES` details all indexes on these tables. This is like a more raw, InnoDB-specific version of metadata also visible through `INFORMATION_SCHEMA.TABLES` and `INFORMATION_SCHEMA.COLUMNS`, but includes internal details only InnoDB knows.
- **INNODB_METRICS:**
  Exposes a large set of internal performance counters (like the number of rows read, buffer pool hits, etc.). These can be turned on or off, reset, and used for performance diagnostics.
- **INNODB_SESSION_TEMP_TABLESPACES:**
  Shows temporary tablespace usage per session, helping to diagnose how sessions use temporary storage.

### Common Patterns

A repeating pattern in the code is:

- Declaring a static `ST_SCHEMA_TABLE` variable or using a function that sets `schema->fields_info`.
- Setting `fill_table` to a static C++ function that fetches data.
- Using macros like `OK(...)` to handle field storage operations. `field->store()` calls are used to store integers, strings, or nulls into the output fields. This is how the code populates rows.
- Ending arrays of `ST_FIELD_INFO` with `END_OF_ST_FIELD_INFO`, a sentinel marker.

Another important aspect is concurrency and synchronization. Many of these functions acquire internal InnoDB mutexes and locks (e.g., `dict_sys_mutex_enter()`) before scanning internal structures. This ensures consistent reads of in-memory data structures.

### Licensing and Authorship

The header comment shows this code is subject to the GNU General Public License (GPL) Version 2. It also clarifies that Oracle and/or its affiliates authored this code for MySQL. The code references OpenSSL and other separately licensed components, explaining the license structure.

### Conclusion

In summary:

- **`i_s.h`**: Declares external references to plugin variables that represent each INFORMATION_SCHEMA table plugin.
- **`i_s.cc`**: Defines the structure and runtime code for a wide range of InnoDB-related INFORMATION_SCHEMA tables. Each table’s columns, initialization logic, and row-filling code is included. When a user queries these tables, the code executes to return the current state of InnoDB’s internal world—transactions, buffer pool contents, indexes, tablespaces, and more.

This code effectively bridges InnoDB’s internal C++ data structures and MySQL’s SQL interface, making complex in-memory information available to DBAs and developers for monitoring, tuning, and troubleshooting InnoDB performance and behavior.