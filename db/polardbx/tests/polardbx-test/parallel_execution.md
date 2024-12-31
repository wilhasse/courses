Below are **two** different approaches—one using a **`CompletionService`**, and one using **`CompletableFuture`** (`anyOf` and `allOf`)—that illustrate how to collect parallel execution results in the order they **finish** or simply wait until **all** are done.

---

## 1) Using `ExecutorCompletionService`
The `ExecutorCompletionService` pattern is straightforward:  
1. You submit a bunch of tasks to the completion service.  
2. You call `completionService.take()`, which **blocks** until *one* of the submitted tasks finishes.  
3. You handle that finished task’s result immediately.  
4. You keep doing `take()` until you have collected all tasks.

**Key point:** You will “receive” results in the order of completion, not in submission order.

```java
private void doChunkedQueryWithCompletionService(SQLSelectStatement originalSelect) throws Exception {
    // Build chunk queries
    List<String> chunks = List.of(
        buildChunkSQL(originalSelect, "< 5000000"),
        buildChunkSQL(originalSelect, ">= 5000000 AND c_custkey < 10000000")
    );

    // Print out chunk info
    System.out.println("Created " + chunks.size() + " chunks:");
    chunks.forEach(sql -> System.out.println("Chunk SQL: " + sql));

    // We will store chunk results here
    List<List<List<Object>>> chunkResults = new ArrayList<>();
    XResult metadataResult = null;

    // Create a CompletionService with your existing ExecutorService
    CompletionService<ChunkResult> completionService = new ExecutorCompletionService<>(executorService);

    // Submit all tasks
    for (String sql : chunks) {
        completionService.submit(() -> {
            // This is what each thread does
            XConnection conn = connectionPool.getNextConnection();
            System.out.println("Starting execution of: " + sql);

            try {
                XResult result = conn.execQuery(sql);
                List<List<Object>> rows = readAllRows(result);
                System.out.println("Finished: " + sql + " with " + rows.size() + " rows");
                return new ChunkResult(result, rows);
            } catch (Exception e) {
                System.out.println("Error executing " + sql + ": " + e.getMessage());
                throw e;
            }
        });
    }

    // Now, gather results in the order they complete
    int received = 0;
    while (received < chunks.size()) {
        System.out.println("Waiting for any chunk to finish...");
        Future<ChunkResult> future = completionService.take(); // blocks until a chunk finishes
        ChunkResult chunkResult = future.get();                // get the actual result

        // Use the first successful chunk to get metadata
        if (metadataResult == null) {
            metadataResult = chunkResult.result;
        }

        chunkResults.add(chunkResult.rows);
        received++;
        System.out.println("Received result for a chunk. Total received: " + received);
    }

    if (metadataResult == null) {
        throw new RuntimeException("No chunk completed successfully to get metadata");
    }

    System.out.println("All chunks completed, merging results...");
    List<List<Object>> mergedRows = mergeChunks(chunkResults);
    System.out.println("Merged " + mergedRows.size() + " total rows");

    sendMergedResponse(metadataResult, mergedRows);
}
```

### Observing the Flow
- Tasks truly run in **parallel** once submitted.  
- The main thread does `completionService.take()` to get whichever chunk finishes first.  
- You’ll see logs in the console reflecting that (e.g., maybe chunk 1 returns before chunk 0).  
- By the time you handle chunk 1, chunk 0 might still be running (or it might also have finished). You don’t lose anything because the `Future` retains the result until you consume it.

---

## 2) Using `CompletableFuture` (with `anyOf` and `allOf`)
If you prefer the Java 8+ “functional” style, or you want non-blocking callbacks, `CompletableFuture` can be very handy. You can also orchestrate in advanced ways (e.g., “collect them in whichever order they finish,” or “wait for all,” etc.).

### a) Waiting for **All** tasks with `CompletableFuture.allOf`
If your goal is to wait until **every chunk** has finished and then process them, you can do:

```java
private void doChunkedQueryWithCompletableFuturesAllOf(SQLSelectStatement originalSelect) {
    List<String> chunks = List.of(
        buildChunkSQL(originalSelect, "< 5000000"),
        buildChunkSQL(originalSelect, ">= 5000000 AND c_custkey < 10000000")
    );

    // We'll collect CompletableFutures here
    List<CompletableFuture<ChunkResult>> futures = new ArrayList<>();

    // Submit each chunk as a supplyAsync to your Executor
    for (String sql : chunks) {
        CompletableFuture<ChunkResult> future = CompletableFuture.supplyAsync(() -> {
            XConnection conn = connectionPool.getNextConnection();
            System.out.println("Starting execution of: " + sql);
            try {
                XResult result = conn.execQuery(sql);
                List<List<Object>> rows = readAllRows(result);
                System.out.println("Finished: " + sql + " with " + rows.size() + " rows");
                return new ChunkResult(result, rows);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }, executorService);

        futures.add(future);
    }

    // allOf returns a new CompletableFuture that completes when all futures complete
    CompletableFuture<Void> allDone = CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));

    // Option 1: block until done, then process results
    allDone.join();  // This blocks the calling thread until everything finishes.

    // Now gather results
    List<List<List<Object>>> chunkResults = new ArrayList<>();
    XResult metadataResult = null;

    for (CompletableFuture<ChunkResult> future : futures) {
        try {
            ChunkResult chunkResult = future.get(); // already done, so this is immediate
            if (metadataResult == null) {
                metadataResult = chunkResult.result;
            }
            chunkResults.add(chunkResult.rows);
        } catch (Exception e) {
            throw new RuntimeException("Error getting chunk result", e);
        }
    }

    // Merge + send
    if (metadataResult == null) {
        throw new RuntimeException("No chunk completed successfully to get metadata");
    }

    List<List<Object>> mergedRows = mergeChunks(chunkResults);
    sendMergedResponse(metadataResult, mergedRows);
}
```

Above, you **do not** need any special loop to check “all the time” if tasks are done. The call to `allDone.join()` (or `allDone.get()`) ensures you only move forward once **all** tasks have finished (or an exception is thrown).

---

### b) Getting tasks in the order they **finish** with `CompletableFuture.anyOf`
If you specifically want to process each chunk **as soon as** it finishes (similar to `CompletionService`), you can do something with `anyOf(...)` in a loop-like fashion—though it’s often easier to just use `thenAccept` or a queue structure for each completion.  

An illustrative (though slightly contrived) approach is:

```java
private void doChunkedQueryWithCompletableFuturesAnyOf(SQLSelectStatement originalSelect) {
    List<String> chunks = List.of(
        buildChunkSQL(originalSelect, "< 5000000"),
        buildChunkSQL(originalSelect, ">= 5000000 AND c_custkey < 10000000")
    );

    // Create a CompletableFuture for each chunk
    List<CompletableFuture<ChunkResult>> futures = new ArrayList<>();
    for (String sql : chunks) {
        CompletableFuture<ChunkResult> cf = CompletableFuture.supplyAsync(() -> {
            XConnection conn = connectionPool.getNextConnection();
            try {
                System.out.println("Executing: " + sql);
                XResult result = conn.execQuery(sql);
                List<List<Object>> rows = readAllRows(result);
                System.out.println("Finished: " + sql);
                return new ChunkResult(result, rows);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }, executorService);
        futures.add(cf);
    }

    // Repeatedly pick off whichever completes first
    List<List<List<Object>>> chunkResults = new ArrayList<>();
    XResult metadataResult = null;

    // We'll move completed futures out of 'futures' as we process them
    while (!futures.isEmpty()) {
        // We can do a single anyOf
        CompletableFuture<?> any = CompletableFuture.anyOf(futures.toArray(new CompletableFuture[0]));

        // Block until *one* future completes
        try {
            Object result = any.get();  // This is the actual ChunkResult from whichever finished first
            if (result instanceof ChunkResult chunkResult) {

                if (metadataResult == null) {
                    metadataResult = chunkResult.result;
                }
                chunkResults.add(chunkResult.rows);

                // Remove it from the list of pending futures
                futures.removeIf(f -> f.isDone());
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // Finally, merge results
    if (metadataResult == null) {
        throw new RuntimeException("No chunk completed successfully to get metadata");
    }
    List<List<Object>> mergedRows = mergeChunks(chunkResults);
    sendMergedResponse(metadataResult, mergedRows);
}
```

**However**, most developers rarely do a raw “`while (!futures.isEmpty()) anyOf(...)`” approach in a real system because `ExecutorCompletionService` is simpler for that pattern. 

If you want a fully **non-blocking** style, you’d typically use `thenAccept` or `thenCompose` or a “reactive” approach, but that introduces a more event-driven design.

---

### When You *Might* Use `anyOf` vs. `allOf`

- **`anyOf`** is handy when you want to react to the *first* of multiple futures that completes (e.g., a “race” condition: you only need the first successful result).
- **`allOf`** is for “wait until everything is done, then proceed.”
- If you want them in “whichever finishes first” order, `CompletionService` is usually simpler, or you can chain `thenAccept` calls.

---

## Do I Need a Constant “Check” Loop?
**No.** You don’t need to spin in a loop checking if tasks are done. With all these APIs, you either:

1. **Block** (`.take()`, `.get()`, `.join()`) until tasks complete.  
2. Or attach **callbacks** (`thenAccept`, `thenApply`, etc.) that run automatically when tasks complete.  

Either way, you’re not polling them in a busy loop. You only do blocking calls or attach callbacks.

---

### TL;DR
- **CompletionService**: best for a straightforward “give me each result as soon as it’s done” in a blocking style.  
- **`CompletableFuture`**:  
  - **`allOf`**: wait for *all tasks* to complete, then gather results.  
  - **`anyOf`**: wait for *any* to complete, possibly in a loop, or do a “race.”  
- None of these patterns lose results, even if one chunk finishes before another. The result is stored in the corresponding `Future`.  