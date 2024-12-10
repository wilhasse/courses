Below is a detailed explanation of what the provided code snippet represents and does. The code is related to MySQL’s InnoDB storage engine and its INFORMATION_SCHEMA (I_S) tables. These specialized tables provide internal details about InnoDB’s internal structures (such as the buffer pool, indexes, tables, and more) to end-users through SQL queries.

------

### Context

InnoDB uses internal system tables (such as `mysql.tables`, `mysql.indexes`, `mysql.columns`, etc.) as well as internal data structures (like the buffer pool) to manage and track metadata related to tablespaces, indexes, columns, and other internal components.

MySQL’s INFORMATION_SCHEMA (I_S) is a schema that provides metadata about database objects. However, by default, it only shows generic metadata. The InnoDB team extends the INFORMATION_SCHEMA with additional tables (through plugins) that expose InnoDB-specific metadata. This code snippet shows the creation, initialization, and data-filling routines for various InnoDB-specific I_S tables.

------

### Overview of the Code

The code snippet includes definitions and initializations of several INFORMATION_SCHEMA tables and their associated plugins:

1. **Common Framework (i_s_common_deinit)**
   This function is a no-op (no operation) placeholder for a common de-initialization routine of dynamic I_S tables.
2. **INNODB_TRX**
   Provides information about InnoDB transactions.
   (Defined earlier in the snippet; shows how code sets up fields and a fill_table function.)
3. **INNODB_CMP and INNODB_CMP_RESET**
   Show compression statistics.
   The code sets fields and uses fill functions to populate these tables with data about compression operations, times, etc.
4. **INNODB_CMP_PER_INDEX and INNODB_CMP_PER_INDEX_RESET**
   Similar to INNODB_CMP but at a per-index granularity.
5. **INNODB_CMPMEM and INNODB_CMPMEM_RESET**
   Provide statistics on the compressed buffer pool memory usage and the number of relocations performed due to compression.
6. **INNODB_METRICS**
   Exposes many internal InnoDB performance metrics.
7. **INNODB_FT_DEFAULT_STOPWORD, INNODB_FT_DELETED, INNODB_FT_BEING_DELETED, INNODB_FT_INDEX_CACHE, INNODB_FT_INDEX_TABLE, INNODB_FT_CONFIG**
   These tables relate to InnoDB Full-Text Search (FTS). They show default stopwords, deleted documents, cached index words, index tables, and configuration parameters used by InnoDB’s FTS subsystem.
8. **INNODB_TEMP_TABLE_INFO**
   Shows information about temporary InnoDB tables currently known to the server, including their IDs, number of columns, and space IDs.
9. **INNODB_BUFFER_POOL_STATS, INNODB_BUFFER_PAGE, INNODB_BUFFER_PAGE_LRU**
   Provide insight into the buffer pool usage:
   - **INNODB_BUFFER_POOL_STATS**: Overall buffer pool statistics (how many pages are modified, how many reads/writes, etc.).
   - **INNODB_BUFFER_PAGE**: Per-page information currently in the buffer pool.
   - **INNODB_BUFFER_PAGE_LRU**: Pages in LRU order.
10. **INNODB_TABLES and INNODB_TABLESTATS**
    - **INNODB_TABLES**: Shows internal metadata about InnoDB tables (like flags, row format, etc.).
    - **INNODB_TABLESTATS**: Provides statistical data about tables, such as the number of rows, cluster index size, modified counters, autoinc, and so forth.
11. **INNODB_INDEXES**
    Metadata and properties of InnoDB indexes, such as index ID, type, and page numbers.
12. **INNODB_COLUMNS** and **INNODB_VIRTUAL**
    - **INNODB_COLUMNS**: Shows column metadata of InnoDB tables (data type, length, default values).
    - **INNODB_VIRTUAL**: Provides information on virtual columns and their base columns.
13. **INNODB_TABLESPACES**
    Displays information about InnoDB tablespaces, their flags, row format, size, space type, and encryption state.
14. **INNODB_CACHED_INDEXES**
    Shows how many pages of each InnoDB index are currently cached in the buffer pool.
15. **INNODB_SESSION_TEMP_TABLESPACES**
    Provides details about session temporary tablespaces created and used by a particular session.

------

### Common Patterns

For each table, the code pattern is generally:

- **Field Definitions (`ST_FIELD_INFO`)**:
  An array of ST_FIELD_INFO structures defines each column's name, length, data type, flags, and other metadata.
- **Init Function**:
  A function (`..._init`) sets the `fields_info` and `fill_table` callback on a `ST_SCHEMA_TABLE` structure. This `fill_table` callback gets called by MySQL to populate the I_S table at query time.
- **Fill Function (`..._fill_table`)**:
  A function that, when invoked by a query to the corresponding I_S table, gathers data (either by scanning InnoDB's internal metadata structures or reading from InnoDB system tables) and uses `schema_table_store_record()` to insert rows into the in-memory I_S table results.
- **Plugins Definition (`struct st_mysql_plugin ...`)**:
  Each table is exposed as an I_S plugin to MySQL. The plugin definition uses `STRUCT_FLD(...)` macros to fill in fields like name, author, description, and importantly, pointers to the init and deinit functions. The plugin is registered with the server, making the I_S table available to queries.

------

### Explanation of How It's Used

When a user issues a query like:

```
sql


Copy code
SELECT * FROM INFORMATION_SCHEMA.INNODB_TABLES;
```

MySQL invokes the plugin responsible for `INNODB_TABLES`. The plugin calls its `fill_table` method (`i_s_innodb_tables_fill_table()` in this case), which then scans the underlying InnoDB data dictionary, extracts the relevant information, and returns it as a result set.

This design allows MySQL and InnoDB developers and DBAs to inspect internal InnoDB metadata easily without resorting to direct file inspection or complicated debug methods.

------

### Summary

The provided code snippet defines multiple INFORMATION_SCHEMA tables that expose internal InnoDB metadata and statistics:

- Transaction info (INNODB_TRX)
- Compression stats (INNODB_CMP, etc.)
- Buffer pool and tablespace details (INNODB_BUFFER_POOL_STATS, INNODB_TABLESPACES)
- Table, index, and column metadata (INNODB_TABLES, INNODB_INDEXES, INNODB_COLUMNS, etc.)
- Session temporary tablespaces (INNODB_SESSION_TEMP_TABLESPACES)
- Full-text search internal tables (INNODB_FT_*)

Each table is implemented as a plugin, making the data available to users via standard SQL queries against the INFORMATION_SCHEMA.