import com.alibaba.druid.sql.ast.statement.SQLSelectStatement;
import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.result.XResult;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class ParallelSplitQueryHandler extends SimpleSplitQueryHandler {
    private final ExecutorService executorService;
    private static final int NUM_THREADS = 4;

    public ParallelSplitQueryHandler(DebugConnection connection) {
        super(connection);
        this.executorService = Executors.newFixedThreadPool(NUM_THREADS);
    }

    @Override
    protected void doChunkedQuery(SQLSelectStatement originalSelect) {
        try {
            List<String> chunks = new ArrayList<>();
            chunks.add(buildChunkSQL(originalSelect, "< 7500"));
            chunks.add(buildChunkSQL(originalSelect, ">= 7500 AND c_custkey < 15000"));
            chunks.add(buildChunkSQL(originalSelect, ">= 15000 AND c_custkey < 22500"));
            chunks.add(buildChunkSQL(originalSelect, ">= 22500"));

            System.out.println("Created " + chunks.size() + " chunks:");
            chunks.forEach(sql -> System.out.println("Chunk SQL: " + sql));

            // Execute all chunks in parallel, but keep track of XResults
            List<Future<ChunkResult>> futures = new ArrayList<>();
            AtomicInteger chunkIndex = new AtomicInteger(0);

            for (String sql : chunks) {
                futures.add(executorService.submit(() -> {
                    int index = chunkIndex.getAndIncrement();
                    System.out.println("Executing chunk " + index + " on thread " +
                            Thread.currentThread().getName());

                    XConnection conn = connectionPool.getNextConnection();
                    try {
                        System.out.println("Starting execution of chunk " + index + ": " + sql);
                        XResult result = conn.execQuery(sql);
                        List<List<Object>> rows = readAllRows(result);
                        System.out.println("Finished chunk " + index + " with " + rows.size() + " rows");
                        return new ChunkResult(result, rows);
                    } catch (Exception e) {
                        System.out.println("Error executing chunk " + index + ": " + e.getMessage());
                        throw new RuntimeException("Error executing chunk: " + sql, e);
                    }
                }));
            }

            // Collect results with timeout
            List<List<List<Object>>> chunkResults = new ArrayList<>();
            XResult metadataResult = null;

            for (int i = 0; i < futures.size(); i++) {
                try {
                    System.out.println("Waiting for chunk " + i + " result...");
                    ChunkResult chunkResult = futures.get(i).get(30, TimeUnit.SECONDS);

                    // Store the first completed XResult for metadata
                    if (metadataResult == null) {
                        metadataResult = chunkResult.result;
                    }

                    chunkResults.add(chunkResult.rows);
                    System.out.println("Received chunk " + i + " result");
                } catch (Exception e) {
                    System.out.println("Error getting chunk " + i + " result: " + e.getMessage());
                    throw new RuntimeException("Error getting chunk result", e);
                }
            }

            if (metadataResult == null) {
                throw new RuntimeException("No chunk completed successfully to get metadata");
            }

            System.out.println("All chunks completed, merging results...");
            List<List<Object>> mergedRows = mergeChunks(chunkResults);
            System.out.println("Merged " + mergedRows.size() + " total rows");

            sendMergedResponse(metadataResult, mergedRows);

        } catch (Exception e) {
            System.out.println("Error in parallel execution: " + e.getMessage());
            e.printStackTrace();
            sendErrorResponse("Error in parallel execution: " + e.getMessage());
        }
    }

    // Helper class to keep XResult and rows together
    private static class ChunkResult {
        final XResult result;
        final List<List<Object>> rows;

        ChunkResult(XResult result, List<List<Object>> rows) {
            this.result = result;
            this.rows = rows;
        }
    }

    private List<List<Object>> mergeChunks(List<List<List<Object>>> chunks) {
        PriorityQueue<ChunkIterator> queue = new PriorityQueue<>(
                (a, b) -> compareRows(a.current(), b.current()));

        // Initialize with first row from each chunk
        for (List<List<Object>> chunk : chunks) {
            if (!chunk.isEmpty()) {
                queue.offer(new ChunkIterator(chunk));
            }
        }

        List<List<Object>> merged = new ArrayList<>();
        while (!queue.isEmpty()) {
            ChunkIterator smallest = queue.poll();
            merged.add(smallest.current());

            if (smallest.hasNext()) {
                smallest.next();
                queue.offer(smallest);
            }
        }

        return merged;
    }

    private static class ChunkIterator {
        private final List<List<Object>> chunk;
        private int index = 0;

        ChunkIterator(List<List<Object>> chunk) {
            this.chunk = chunk;
        }

        List<Object> current() {
            return chunk.get(index);
        }

        boolean hasNext() {
            return index < chunk.size() - 1;
        }

        void next() {
            index++;
        }
    }

    private int compareRows(List<Object> row1, List<Object> row2) {
        // Compare based on first column
        long key1 = Long.parseLong(String.valueOf(row1.get(0)));
        long key2 = Long.parseLong(String.valueOf(row2.get(0)));
        return Long.compare(key1, key2);
    }

    public void close() {
        try {
            executorService.shutdown();
            if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
        }
        // Then call parent's close to clean up connections
        super.close();
    }
}