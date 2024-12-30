Below is a detailed explanation of how the **ParallelSplitServer** and **ParallelSplitQueryHandler** classes work, building on top of the logic from the earlier split-server example:

---

## 1. ParallelSplitServer

### a. Inheritance from SimpleServer

`ParallelSplitServer` extends `SimpleServer`, which means it reuses all of `SimpleServer`’s standard networking and server-bootstrapping logic:

- It sets up `NIOProcessor` threads,
- Creates the `NIOAcceptor` on a given port,
- Initializes timers, etc.

The primary difference is **which query handler** it attaches to new connections.

### b. Overridden `createConnectionFactory()`

Similar to the earlier “split” version, `ParallelSplitServer` overrides the `createConnectionFactory()` method to return its own `DebugSplitConnectionFactory`:

```java
@Override
protected FrontendConnectionFactory createConnectionFactory() {
    return new DebugSplitConnectionFactory();
}
```

### c. DebugSplitConnectionFactory inner class

Inside `ParallelSplitServer`, there is an inner class `DebugSplitConnectionFactory`. In `getConnection(...)`:

1. It **creates a** new `DebugConnection`.
2. **Sets** its query handler to a `ParallelSplitQueryHandler` **instead** of the original or simple split handler.

That means that any time a client connects, each `DebugConnection` will use the parallel logic for splitting queries:

```java
class DebugSplitConnectionFactory extends FrontendConnectionFactory {
    @Override
    protected FrontendConnection getConnection(SocketChannel channel) {
        System.out.println("Creating new connection for channel: " + channel);
        DebugConnection c = new DebugConnection(channel);
        c.setQueryHandler(new ParallelSplitQueryHandler(c)); // <--- parallel splitting
        return c;
    }
}
```

### d. Main method

The `main()` method is straightforward: it calls `getInstance().startup()`, then loops forever until the server is shut down. The difference from `SimpleServer` is purely in the query handler being used.

---

## 2. ParallelSplitQueryHandler

`ParallelSplitQueryHandler` is a subclass of `SimpleSplitQueryHandler`. It inherits all the general logic for:

1. **Parsing** incoming SQL.
2. **Determining** if it’s chunkable (`canChunk(...)`).
3. **Sending** results to the client using the MySQL wire protocol.

However, it overrides **`doChunkedQuery(...)`** to split the queries in parallel across multiple threads.

### a. ExecutorService for parallel tasks

In the constructor, `ParallelSplitQueryHandler` creates an `ExecutorService` (fixed thread pool) with a fixed size (default 4 threads in this example):

```java
public ParallelSplitQueryHandler(DebugConnection connection) {
    super(connection);
    this.executorService = Executors.newFixedThreadPool(NUM_THREADS);
}
```

This allows the handler to execute multiple sub-queries simultaneously.

### b. Overridden `doChunkedQuery(...)`

Rather than generating **two** chunks as in `SimpleSplitQueryHandler`, this method generates **four** sub-queries, each with a different range condition:

```java
List<String> chunks = new ArrayList<>();
chunks.add(buildChunkSQL(originalSelect, "< 7500"));
chunks.add(buildChunkSQL(originalSelect, ">= 7500 AND c_custkey < 15000"));
chunks.add(buildChunkSQL(originalSelect, ">= 15000 AND c_custkey < 22500"));
chunks.add(buildChunkSQL(originalSelect, ">= 22500"));
```

So effectively, it’s splitting the table into 4 segments (0–7499, 7500–14999, 15000–22499, 22500+). Then it submits tasks for each of these chunks into the `executorService`.

#### i. Submitting each chunk to the thread pool

It loops over these chunked SQL statements, and for each one, it calls:

```java
List<Future<ChunkResult>> futures = new ArrayList<>();

futures.add(executorService.submit(() -> {
    // This code runs in a separate thread
    XConnection conn = connectionPool.getNextConnection();
    XResult result = conn.execQuery(sql);
    List<List<Object>> rows = readAllRows(result);
    return new ChunkResult(result, rows);
}));
```

The logic is:

1. Get the next available connection from `connectionPool`.
2. Execute the chunk’s SQL (`execQuery(sql)`).
3. Read **all** rows from that result into a `List<List<Object>>`.
4. Return a small helper class `ChunkResult` that holds:
   - The `XResult` (for metadata)
   - The list of rows read in memory

Each chunk runs **in parallel** because each chunk is submitted to the ExecutorService.

#### ii. Collecting futures with timeouts

After submitting all chunks, `ParallelSplitQueryHandler` calls `futures.get(i).get(30, TimeUnit.SECONDS)` on each future, waiting up to 30 seconds for it to finish. If a chunk fails or times out, it throws an exception. Otherwise, it collects the chunk’s data:

```java
ChunkResult chunkResult = futures.get(i).get(30, TimeUnit.SECONDS);

if (metadataResult == null) {
    metadataResult = chunkResult.result;  // store first chunk's XResult for metadata
}

chunkResults.add(chunkResult.rows);
```

Here, `metadataResult` is used later for constructing the column definitions in the final response.

### c. Merging chunk results

Once all chunk results are collected, it merges them. Because there are multiple chunks, we need to **sort** the combined rows (based on a key column) into a single ordered stream. That logic is in `mergeChunks(...)`.

#### i. PriorityQueue merging

`mergeChunks(List<List<List<Object>>> chunks)` method uses a **`PriorityQueue`** of `ChunkIterator`. Each `ChunkIterator` points to one chunk (list of lists). The `PriorityQueue` is ordered by a custom comparator— in this example, it compares the first column (assuming it’s numeric):

```java
PriorityQueue<ChunkIterator> queue = new PriorityQueue<>(
    (a, b) -> compareRows(a.current(), b.current())
);
```

The method:

1. Inserts each chunk’s first row into the queue.
2. Pops the smallest row (by the first column).
3. Adds it to the final `merged` list.
4. Advances that chunk’s iterator by one.
5. Re-inserts that chunk into the queue (if more rows remain).

Essentially, it’s a multi-way merge (like merging sorted files). In the end, we get one sorted list of all rows from the parallel chunks.

### d. Sending the merged response

Just like in `SimpleSplitQueryHandler`, once we have a final merged `List<List<Object>>` plus one chunk’s `XResult` for metadata, we call:

```java
sendMergedResponse(metadataResult, mergedRows);
```

This writes out a single MySQL wire-protocol result set to the client:
1. **ResultSetHeader** (how many columns)
2. **Field packets** (column definitions)
3. **(EOF)**
4. **RowData** for each merged row
5. **(EOF)**

Thus, the client sees a single, contiguous, fully sorted result set, but behind the scenes, it was fetched in parallel from multiple range queries.

### e. Shutdown / close

In `close()`, the handler shuts down the `ExecutorService` properly, waiting for any in-progress tasks to finish, then calling `super.close()` to clean up the rest of the resources.

---

## Summary

1. **ParallelSplitServer**:  
   - Extends `SimpleServer`.  
   - Provides a custom `DebugSplitConnectionFactory` that creates `DebugConnection` objects with a `ParallelSplitQueryHandler`.

2. **ParallelSplitQueryHandler**:  
   - Extends `SimpleSplitQueryHandler` but overrides `doChunkedQuery(...)` to split queries into **four** range chunks.  
   - Submits these chunks to a fixed-thread-pool ExecutorService for **parallel execution**.  
   - Collects the results, merges them via a **priority queue** for sorted output, and sends them as a single result set.  

3. **Key differences from SimpleSplit**:
   - The use of an **ExecutorService** to run chunk queries in **parallel**.  
   - The merging logic is a **multi-way merge** with a priority queue to handle multiple chunks.  
   - Graceful shutdown of the thread pool in the `close()` method.  

Overall, these classes demonstrate how to split queries into parallel chunks, then gather & merge the results to the client as a single logical result set.