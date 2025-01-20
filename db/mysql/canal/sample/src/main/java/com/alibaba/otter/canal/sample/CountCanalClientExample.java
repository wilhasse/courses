package com.alibaba.otter.canal.sample;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;

import com.alibaba.otter.canal.client.CanalConnectors;
import com.alibaba.otter.canal.client.CanalConnector;
import com.alibaba.otter.canal.protocol.Message;
import com.alibaba.otter.canal.protocol.CanalEntry.Column;
import com.alibaba.otter.canal.protocol.CanalEntry.Entry;
import com.alibaba.otter.canal.protocol.CanalEntry.EntryType;
import com.alibaba.otter.canal.protocol.CanalEntry.EventType;
import com.alibaba.otter.canal.protocol.CanalEntry.RowChange;
import com.alibaba.otter.canal.protocol.CanalEntry.RowData;

public class CountCanalClientExample {

    // Map to track the count of INSERT+UPDATE rows per table.
    private static final Map<String, Long> tableRowsAffectedMap = new HashMap<>();

    // We will print our stats once every minute (60 seconds).
    private static final long PRINT_INTERVAL = 60_000;
    private static long lastPrintTime = System.currentTimeMillis();

    public static void main(String[] args) {

        CanalConnector connector = CanalConnectors.newSingleConnector(
            new InetSocketAddress("10.200.15.21", 11111),
            "example",
            "",
            ""
        );

        int batchSize = 1000;
        int emptyCount = 0;
        int totalEmptyCount = 120;

        try {
            connector.connect();
            // Subscribe to all tables in all schemas. Or narrow as needed.
            connector.subscribe(".*\\..*");
            connector.rollback();

            while (emptyCount < totalEmptyCount) {
                Message message = connector.getWithoutAck(batchSize);
                long batchId = message.getId();
                int size = message.getEntries().size();

                if (batchId == -1 || size == 0) {
                    emptyCount++;
                    System.out.println("empty count : " + emptyCount);
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        // Handle interruption
                    }
                } else {
                    emptyCount = 0;
                    processEntries(message.getEntries());
                }
                connector.ack(batchId);

                // Check if it's time to print the top 10.
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastPrintTime >= PRINT_INTERVAL) {
                    printTop10Tables();
                    // Reset the map
                    tableRowsAffectedMap.clear();
                    // Reset the last print time
                    lastPrintTime = currentTime;
                }
            }

            System.out.println("empty too many times, exit");
        } finally {
            connector.disconnect();
        }
    }

    /**
     * Process the row change entries, and update the 'tableRowsAffectedMap'
     * accordingly (INSERT + UPDATE).
     */
    private static void processEntries(List<Entry> entries) {
        for (Entry entry : entries) {
            if (entry.getEntryType() == EntryType.TRANSACTIONBEGIN
                    || entry.getEntryType() == EntryType.TRANSACTIONEND) {
                continue;
            }

            RowChange rowChange;
            try {
                rowChange = RowChange.parseFrom(entry.getStoreValue());
            } catch (Exception e) {
                throw new RuntimeException(
                    "ERROR parsing event data: " + entry.toString(), e
                );
            }

            EventType eventType = rowChange.getEventType();
            String schemaName = entry.getHeader().getSchemaName();
            String tableName = entry.getHeader().getTableName();

            // Count how many rows in this event (each RowData is one row).
            int rowCount = rowChange.getRowDatasList().size();

            // For this example, only count INSERT + UPDATE.
            if (eventType == EventType.INSERT || eventType == EventType.UPDATE) {
                // Increase the count for this table by 'rowCount'
                String key = schemaName + "." + tableName;
                tableRowsAffectedMap.merge(key, (long) rowCount, Long::sum);
            }
        }
    }

    /**
     * Print the top 10 tables by rows affected (INSERT+UPDATE).
     */
    private static void printTop10Tables() {
        if (tableRowsAffectedMap.isEmpty()) {
            System.out.println("No rows affected in the last minute.");
            return;
        }

        System.out.println("=== Top 10 tables (by rows INSERT/UPDATE) in the last minute ===");
        // Sort descending by number of affected rows, then take top 10
        List<Map.Entry<String, Long>> top10 = tableRowsAffectedMap.entrySet()
            .stream()
            .sorted((e1, e2) -> Long.compare(e2.getValue(), e1.getValue()))
            .limit(10)
            .collect(Collectors.toList());

        for (Map.Entry<String, Long> entry : top10) {
            System.out.println(String.format("Table: %s -> %d rows", entry.getKey(), entry.getValue()));
        }
        System.out.println("=================================================================");
    }
}
