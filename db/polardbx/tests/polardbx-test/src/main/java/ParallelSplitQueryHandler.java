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
            chunks.add(buildChunkSQL(originalSelect, "< 5"));
            chunks.add(buildChunkSQL(originalSelect, ">= 5 AND c_custkey < 10"));
            chunks.add(buildChunkSQL(originalSelect, ">= 10 AND c_custkey < 15"));
            chunks.add(buildChunkSQL(originalSelect, ">= 15"));

            System.out.println("Created " + chunks.size() + " chunks:");
            for (String sql : chunks) {
                System.out.println("Chunk SQL: " + sql);
            }

            List<Future<List<List<Object>>>> futures = new ArrayList<>();
            AtomicInteger chunkIndex = new AtomicInteger(0); // To track chunk execution

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
                        return rows;
                    } catch (Exception e) {
                        System.out.println("Error executing chunk " + index + ": " + e.getMessage());
                        throw new RuntimeException("Error executing chunk: " + sql, e);
                    }
                }));
            }

            System.out.println("All chunks submitted, waiting for results...");

            // Collect results with timeout
            List<List<List<Object>>> chunkResults = new ArrayList<>();
            for (int i = 0; i < futures.size(); i++) {
                try {
                    System.out.println("Waiting for chunk " + i + " result...");
                    chunkResults.add(futures.get(i).get(30, TimeUnit.SECONDS));
                    System.out.println("Received chunk " + i + " result");
                } catch (Exception e) {
                    System.out.println("Error getting chunk " + i + " result: " + e.getMessage());
                    throw new RuntimeException("Error getting chunk result", e);
                }
            }

            System.out.println("All chunks completed, merging results...");

            // Merge results using priority queue
            List<List<Object>> mergedRows = mergeChunks(chunkResults);
            System.out.println("Merged " + mergedRows.size() + " total rows");

            // Send response using metadata from first chunk
            System.out.println("Getting metadata using first chunk query");
            XResult metadataResult = connectionPool.getNextConnection()
                    .execQuery(chunks.get(0));
            sendMergedResponse(metadataResult, mergedRows);

        } catch (Exception e) {
            System.out.println("Error in parallel execution: " + e.getMessage());
            e.printStackTrace();
            sendErrorResponse("Error in parallel execution: " + e.getMessage());
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