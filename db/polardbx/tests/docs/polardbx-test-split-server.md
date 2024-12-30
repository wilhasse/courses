Below is an overview of how the **SimpleSplitServer** and **SimpleSplitQueryHandler** classes work together to intercept incoming SQL queries, determine whether they can be “chunked,” run multiple sub-queries, merge the results, and finally return a single combined result set to the client.

------

## 1. SimpleSplitServer

### a. Extends SimpleServer

`SimpleSplitServer` extends `SimpleServer`, meaning it reuses most of the server initialization logic (creating NIO processors, timers, etc.).

### b. Overridden `createConnectionFactory()`

The main difference is that it overrides the `createConnectionFactory()` method to return its own factory:

```java
@Override
protected FrontendConnectionFactory createConnectionFactory() {
    return new DebugSplitConnectionFactory();
}
```

### c. DebugSplitConnectionFactory

Inside `SimpleSplitServer`, there's an inner class `DebugSplitConnectionFactory` that extends `FrontendConnectionFactory`. It overrides `getConnection(SocketChannel channel)` to:

1. Create a new `DebugConnection`.
2. **Crucially**, set its query handler to a `SimpleSplitQueryHandler`, instead of the normal `SimpleQueryHandler`.

```java
@Override
protected FrontendConnection getConnection(SocketChannel channel) {
    System.out.println("Creating new connection for channel: " + channel);
    DebugConnection c = new DebugConnection(channel);
    c.setQueryHandler(new SimpleSplitQueryHandler(c)); // <--- using the split handler
    return c;
}
```

### d. Main method

The `main()` method in `SimpleSplitServer` is practically the same as in `SimpleServer`: it starts up, listens on the port, and waits for queries. The only difference is that all new connections get `SimpleSplitQueryHandler`.

------

## 2. SimpleSplitQueryHandler

`SimpleSplitQueryHandler` is a subclass of `SimpleQueryHandler` and is where the “split” logic lives.

### a. `query(String sql)` override

This method is called whenever a query arrives from the client. The flow is:

1. **Parse** the SQL using Druid (`MySqlStatementParser`).
2. Check if it is **chunkable** (a single `SELECT` on table “customer” with an `ORDER BY c_name`).
3. If it is chunkable, do a **two-chunk** query in `doChunkedQuery()`.
4. Otherwise (fallback), just execute the query directly.

Pseudocode from `query(String sql)`:

```java
public void query(String sql) {
    // 1) Parse with Alibaba Druid
    SQLStatement stmt;
    try {
        MySqlStatementParser parser = new MySqlStatementParser(sql);
        List<SQLStatement> stmtList = parser.parseStatementList();
        ...
        stmt = stmtList.get(0);
    } catch (ParserException pe) {
        sendErrorResponse("SQL parse error: " + pe.getMessage());
        return;
    }

    // 2) Check if chunkable
    if (stmt instanceof SQLSelectStatement) {
        SQLSelectStatement selectStmt = (SQLSelectStatement) stmt;
        if (canChunk(selectStmt)) {
            doChunkedQuery(selectStmt);
            return;
        }
    }

    // 3) Fallback if not chunkable
    try {
        XResult result = connectionPool.getNextConnection().execQuery(sql);
        sendResultSetResponse(result);
    } ...
}
```

### b. Checking if a query is chunkable

The logic in `canChunk(...)` is basically:

1. Is it a single `SQLSelectQueryBlock` (i.e., not a compound query, not multiple SELECTs, not subqueries)?
2. Is the `FROM` table named `customer`?
3. Is there an `ORDER BY` with exactly one column, `c_name`?

Only if all these pass do we say “yes, we can chunk this.”

### c. Doing a chunked query

`doChunkedQuery(SQLSelectStatement originalSelect)` shows the gist of how to split a single logical query into two sub-queries:

1. **Rewrite** the original SQL statement into two “chunk” statements: one with `WHERE c_custkey < 10` and one with `WHERE c_custkey >= 10`.

   ```java
   String sqlChunk1 = buildChunkSQL(originalSelect, "< 10");
   String sqlChunk2 = buildChunkSQL(originalSelect, ">= 10");
   ```

2. **Execute** each chunk separately— in a real scenario, these might go to different nodes or connections.

   ```java
   XResult result1 = connectionPool.getNextConnection().execQuery(sqlChunk1);
   XResult result2 = connectionPool.getNextConnection().execQuery(sqlChunk2);
   ```

3. **Merge** the results in memory. Here, because the user wants the results sorted on `c_name` (or some column), the code does a simple 2-way merge (`mergeOrderedResults(result1, result2)`) by reading rows from each result set, comparing the key column, and building a merged list.

4. **Send** the merged results as one unified result set back to the client in `sendMergedResponse(...)`.

### d. Building the chunked SQL

`buildChunkSQL(...)` tries to insert the extra condition (`AND c_custkey < 10` or `AND c_custkey >= 10`) just before the `ORDER BY`. If there’s no `ORDER BY`, it simply appends the condition at the end. If there’s no `WHERE` clause, it creates one.

### e. Merging results

`mergeOrderedResults(result1, result2)`:

1. Reads **all** rows from each `XResult` into two separate lists (`rows1` and `rows2`).
2. Does a **2-way merge** based on the first column in each row, comparing them.
3. Returns a single merged list (`merged`).

### f. Sending merged response

Once we have the merged rows, we can’t simply reuse one original `XResult` because that’s a streaming mechanism. Instead, we manually construct a MySQL protocol response:

1. **Write a `ResultSetHeaderPacket`** indicating how many columns there are.
2. **Write each `FieldPacket`** describing each column (column name, type, etc.).
3. **Send an `EOFPacket`** (if required by the protocol).
4. **Send each row** in a `RowDataPacket`.
5. **Send a final `EOFPacket`**.

This way, the client sees one contiguous result set, even though it was built from multiple sub-queries.

------

## Putting it all together

- **SimpleSplitServer** simply creates a different `FrontendConnectionFactory` that uses `SimpleSplitQueryHandler`.

- SimpleSplitQueryHandler

   is identical to 

  ```
  SimpleQueryHandler
  ```

   in structure but overrides the 

  ```
  query(...)
  ```

   method to:

  1. Check if the query can be split (“chunked”) into multiple range queries.
  2. If yes, run each sub-query, merge them, and send a single combined result set back to the client.
  3. If not, fall back to executing the query normally.

This design allows you to demonstrate how to intercept, rewrite, split, merge, and present queries in a single logical result—even though you might be pulling data from multiple sources or multiple chunks behind the scenes.