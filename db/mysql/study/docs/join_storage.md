# Joins and MySQL Storage Engine

**Short Summary**:  
In MySQL’s query execution engine, joins are processed using a "volcano iterator" model, where each step of the execution plan requests the "next" row from one or more iterators (tables or intermediate result sets) until no more rows match. The storage engine (such as the MEMORY engine) provides a handler interface that the MySQL server layer calls to retrieve rows incrementally. When the optimizer chooses a join order and method, the executor uses these handler calls (like `rnd_init()`, `rnd_next()`, `index_init()`, `index_read()`, `index_next()`) to fetch rows from each underlying table. The volcano iterator concept maps directly onto repeated calls to the handler’s row-fetching methods. Essentially, each table in the join appears as an iterator that can produce rows on-demand, and the join code orchestrates these iterators together to implement nested loops or other join strategies.

----------

**Detailed Explanation**:

1.  **Volcano Iterator Model in MySQL**:  
    MySQL’s query execution can be conceptualized using the volcano iterator model (also known as the iterator execution model). In this model, each node in the query execution plan is an iterator that supports a "next row" interface. A join is represented as a tree of iterators:
    
    -   Base tables are at the leaves, producing rows one at a time via handler methods.
    -   Higher-level nodes combine or filter these rows (for example, performing JOIN operations, grouping, or sorting).
    
    When you run a query that involves multiple tables, MySQL’s join executor sets up a sequence of nested loops or other join algorithms. Each "loop" or join step will call into a table’s iterator to get rows. Internally, these iterators correspond to calls through the handler interface that the storage engine (in this case, the MEMORY/HEAP engine) provides.
    
2.  **Handler Interface and the Storage Engine**:  
    The code snippet you have is part of the MySQL MEMORY storage engine’s `handler` class, `ha_heap`. This class implements a standard set of methods that the MySQL server calls to perform operations on a table:
    
    -   `rnd_init()` and `rnd_next()` for table scans.
    -   `index_init()`, `index_read()`, `index_next()` (and related) for indexed lookups.
    -   `write_row()`, `update_row()`, and `delete_row()` for modifying data.
    
    The MySQL server doesn’t directly iterate over the storage engine’s data structures. Instead, it calls these handler methods to:
    
    -   Start a table scan or an index lookup (`rnd_init()` or `index_init()`).
    -   Fetch rows one by one (`rnd_next()` for full table scans, `index_next()` or `index_read()` for indexed queries).
    -   Position to specific keys, reset cursors, and so forth.
3.  **Relating Volcano Iterators to the Handler Methods in a Join**:  
    Consider a simple join query:
    
    ```sql
    SELECT * 
    FROM t1 
    JOIN t2 ON t1.col = t2.col 
    WHERE t1.col > 10;
    
    ```
    
    Let’s say `t1` and `t2` are MEMORY tables. The execution might proceed as follows:
    
    -   The optimizer decides on a join order. For example, it chooses to read from `t1` first and for each row in `t1`, find matching rows in `t2`.
        
    -   The executor begins by calling:
        
        ```c++
        t1->file->rnd_init(true);
        
        ```
        
        This prepares a full table scan on `t1`. Under the hood, `rnd_init()` calls `heap_scan_init(file)` to initialize an iterator for scanning all rows in `t1`.
        
    -   To fetch rows from `t1`, the executor repeatedly calls:
        
        ```c++
        while (!t1->file->rnd_next(t1->record[0])) {
          // Process row from t1
        }
        
        ```
        
        Internally, `rnd_next()` calls `heap_scan(file, buf)` to get the next row from `t1`. Each call returns the next row until no more rows are available.
        
    -   For each row retrieved from `t1`, suppose the server wants to find matching rows in `t2` based on `t1.col = t2.col`. If `t2` has an index on `col`, the executor will:
        
        ```c++
        t2->file->index_init(index_number, false);
        // Construct a search key from t1.col
        t2->file->index_read_map(t2->record[0], key_buffer, key_map, HA_READ_KEY_EXACT);
        
        ```
        
        Here, `index_init()` prepares `t2`’s index for traversal. `index_read_map()` uses the given search key (derived from t1’s current row) to find matching rows in `t2`. If a row is found, the executor processes it. It may then call `index_next()` on `t2` to find additional matching rows.
        
    -   The interaction follows a nested loop:
        
        -   Outer loop: Each call to `rnd_next()` on `t1` gives a new `t1` row.
        -   Inner loop: For each `t1` row, calls `index_read_map()` and possibly `index_next()` on `t2` to find all matching rows.
    
    This pattern effectively implements the volcano model: the join node calls its child iterators (which correspond to handler calls on storage engines) to produce rows. Each "next row" request percolates down the execution tree until a storage engine method returns an actual row.
    
4.  **Practical Example**:
    
    Let's say you have the following MEMORY tables and query:
    
    ```sql
    CREATE TABLE t1 (
      id INT,
      val INT,
      INDEX(val)
    ) ENGINE=MEMORY;
    
    CREATE TABLE t2 (
      id INT,
      val INT,
      INDEX(val)
    ) ENGINE=MEMORY;
    
    INSERT INTO t1 VALUES (1,10),(2,20),(3,30);
    INSERT INTO t2 VALUES (100,10),(200,20),(300,20);
    
    SELECT t1.id, t2.id
    FROM t1
    JOIN t2 ON t1.val = t2.val;
    
    ```
    
    **What happens in the engine?**
    
    -   MySQL optimizer chooses to do a table scan on `t1` and use index lookups on `t2`.
    
    Steps:
    
    1.  **Initialize `t1` scan**: `t1->file->rnd_init(true)` -> `heap_scan_init(file_for_t1)`.
    2.  **Get rows from `t1`**: Call `t1->file->rnd_next(t1_buf)` repeatedly.
        -   1st call returns `(id=1, val=10)`
        -   2nd call returns `(id=2, val=20)`
        -   3rd call returns `(id=3, val=30)`
        -   Then returns EOF.
    
    For each row from `t1`:
    
    -   Construct a key = `val` from the current `t1` row.
    -   **Initialize `t2` index**: `t2->file->index_init(val_index, false)`.
    -   **Index lookup in `t2`**: `t2->file->index_read_map(t2_buf, key=(val), ... , HA_READ_KEY_EXACT)`.
    
    When processing `(id=1, val=10)` from `t1`:
    
    -   Look up `val=10` in `t2`:
        -   `index_read_map()` finds `(id=100, val=10)`.
        -   Return it to the join executor.
        -   `index_next()` is called to see if there are more matches for `val=10`.
        -   No more matches, move on.
    
    For `(id=2, val=20)` from `t1`:
    
    -   `index_read_map()` finds `(id=200, val=20)`.
    -   `index_next()` finds `(id=300, val=20)` because `t2` can have multiple rows with `val=20`.
    -   After that, no more matches.
    
    For `(id=3, val=30)` from `t1`:
    
    -   `index_read_map()` attempts to find `val=30` in `t2`, no match found, no rows returned.
    
    In the end, the join iterator returns rows:
    
    -   `(1,100)`, `(2,200)`, `(2,300)` as the result of the join.
    
    Notice how each call to `rnd_next()` or `index_read_map()` is essentially the volcano "next" operation at the storage engine level. The join executor just orchestrates these calls and combines their results.
    

# B+TREE

**Short Summary**:  
MySQL often uses indexes (such as B+Trees) on join columns to speed up lookup operations when joining tables. During nested-loop joins, the inner table lookups are typically performed as ordered searches on these indexes. This results in `O(log N)` complexity per lookup rather than scanning the entire table. MySQL also uses various optimizations (like Batched Key Access or Multi-Range Read) to reduce random I/O and improve overall join performance. In the MEMORY engine’s case, indexes are kept in RAM, making lookups very fast, but still reliant on well-chosen indexing strategies for optimal performance.

----------

**Detailed Explanation**:

1.  **Nested-Loop Joins and Index Lookup**:  
    The common join algorithm in MySQL is a nested-loop join:
    
    -   MySQL picks one table as the "outer" table. Rows are retrieved from this table one by one (often via a sequential scan).
    -   For each row from the outer table, MySQL uses the join condition and tries to find matching rows in the "inner" table. If the inner table has an appropriate index on the join column, MySQL uses that index to quickly look up matching rows instead of scanning the entire inner table.
    
    If the inner table is indexed on the join column, this lookup becomes an `O(log N)` operation due to the B+Tree index. Without the index, you might end up with a full table scan, which is `O(N)`.
    
2.  **Ordered Searches in a B+Tree**:  
    When the join condition involves equality on a column that’s indexed (e.g., `t2.val = t1.val`), MySQL can use the index on `t2.val`. A B+Tree index is sorted, allowing for binary-search-like operations:
    
    -   **Index Read**: MySQL uses functions like `index_read_map()` (as you saw in the MEMORY handler code) to find the first record matching a given key. Because the B+Tree is balanced and ordered, finding a particular key involves descending the tree layers in `O(log N)` time.
    -   **Index Next**: Once a matching entry is found, MySQL may use `index_next()` to iterate through subsequent entries in key order, which is efficient if multiple rows match the same key or range.
    
    Thus, for equality joins on indexed columns, MySQL effectively performs an ordered search in the B+Tree structure.
    
3.  **Performance Considerations**:
    
    -   **Index Quality & Cardinality**: The performance benefits come from choosing the right indexes. A well-chosen index with good selectivity on the join column means fewer rows are fetched from the inner table, leading to much faster joins.
        
    -   **In-Memory Structures vs. On-Disk**: The MEMORY engine stores both data and indexes entirely in RAM. While a B+Tree-based index in a MEMORY table still offers `O(log N)` complexity, in-memory lookups are extremely fast due to reduced I/O overhead. For on-disk engines like InnoDB, random lookups can be more expensive because they may involve disk I/O. In that case, MySQL tries to optimize by caching frequently used index pages and rows.
        
    -   **Batched Key Access & Multi-Range Read**: For certain queries, the optimizer may choose strategies like:
        
        -   **Batched Key Access (BKA)**: It retrieves batches of join keys from the outer table and then does lookups in a more "batched" manner, potentially reducing random disk reads.
        -   **Multi-Range Read (MRR)**: Allows reading ranges from an index in a more I/O-friendly sequence, reducing random accesses on disk.
    
    These optimizations improve throughput by minimizing the number of random I/O operations, turning them into more sequential reads, which are faster.
    
4.  **Memory Engine Specifics**:  
    In the MEMORY engine:
    
    -   Indexes can be either hash-based or tree-based. For B-Tree indexes, lookups are fast since everything is in RAM.
    -   Hash indexes can provide `O(1)` average lookup time, which can be even faster for equality joins but don’t preserve ordering as well as B+Trees. If a B-Tree index is used, it provides ordered access and efficient lookups.
    
    Because all data is in memory, one major bottleneck—disk I/O—is eliminated, making join lookups very fast. However, the complexity (logarithmic for B+Trees) still matters when scaling to very large tables in memory.