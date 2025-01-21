package com.alibaba.otter.canal.sample;

import java.net.InetSocketAddress;
import java.util.*;
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

    // Holds INSERT, UPDATE, and DELETE counts for each table.
    private static final Map<String, TableStats> tableStatsMap = new HashMap<>();

    // Print interval (60 seconds).
    private static final long PRINT_INTERVAL = 60_000;
    private static long lastPrintTime = System.currentTimeMillis();

    // Thresholds for update and delete alarms.
    private static final long UPDATE_THRESHOLD = 5000;  // Example threshold
    private static final long DELETE_THRESHOLD = 1000;  // Example threshold

    public static void main(String[] args) {
        // Check if IP address is provided as command line argument
        if (args.length < 1) {
            System.out.println("Please provide IP address as command line argument");
            System.out.println("Usage: java YourClassName <ip_address>");
            return;
        }

        String ipAddress = args[0];
        CanalConnector connector = CanalConnectors.newSingleConnector(
            new InetSocketAddress(ipAddress, 31111),
            "percona",
            "",
            ""
        );

        int batchSize = 1000;
        int emptyCount = 0;
        int totalEmptyCount = 120;

        try {
            connector.connect();
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

                long currentTime = System.currentTimeMillis();
                if (currentTime - lastPrintTime >= PRINT_INTERVAL) {
                    printStatsAndCheckAlarms();
                    tableStatsMap.clear();
                    lastPrintTime = currentTime;
                }
            }
            System.out.println("empty too many times, exit");
        } finally {
            connector.disconnect();
        }
    }

    /**
     * Process the row change entries, and update the TableStats
     * (insertCount, updateCount, deleteCount) accordingly.
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

            // Count how many rows in this event (each RowData = one row).
            int rowCount = rowChange.getRowDatasList().size();

            // Get or create the TableStats object for this table.
            String key = schemaName + "." + tableName;
            TableStats stats = tableStatsMap.computeIfAbsent(key, k -> new TableStats());

            // Update counts individually.
            if (eventType == EventType.INSERT) {
                stats.insertCount += rowCount;
            } else if (eventType == EventType.UPDATE) {
                stats.updateCount += rowCount;
            } else if (eventType == EventType.DELETE) {
                stats.deleteCount += rowCount;
            }
        }
    }

    /**
     * Print the stats for each table and then check whether any table
     * has an updateCount or deleteCount above our thresholds.
     */
    private static void printStatsAndCheckAlarms() {
        if (tableStatsMap.isEmpty()) {
            System.out.println("No rows affected in the last minute.");
            return;
        }

        System.out.println("=== Tables (INSERT/UPDATE/DELETE counts) in the last minute ===");
        // Sort tables by total row changes (insert + update + delete) descending,
        // just for a sample display. You could also separately sort if needed.
        List<Map.Entry<String, TableStats>> sorted = tableStatsMap.entrySet().stream()
            .sorted((e1, e2) -> Long.compare(e2.getValue().totalChanges(), e1.getValue().totalChanges()))
            .collect(Collectors.toList());

        // Print stats for each table
        sorted.forEach(entry -> {
            String table = entry.getKey();
            TableStats stats = entry.getValue();
            System.out.println(String.format("Table: %s -> INSERT: %d, UPDATE: %d, DELETE: %d",
                table, stats.insertCount, stats.updateCount, stats.deleteCount));
        });
        System.out.println("=================================================================");

        // Now check alarms for the largest update count and largest delete count.
        checkAlarmForUpdates(sorted);
        checkAlarmForDeletes(sorted);
    }

    /**
     * Find the table with the maximum UPDATE count, trigger alarm if over threshold.
     */
    private static void checkAlarmForUpdates(List<Map.Entry<String, TableStats>> sortedEntries) {
        Optional<Map.Entry<String, TableStats>> maxUpdateEntry = sortedEntries.stream()
            .max(Comparator.comparingLong(e -> e.getValue().updateCount));
        if (maxUpdateEntry.isPresent()) {
            long maxUpdateCount = maxUpdateEntry.get().getValue().updateCount;
            String table = maxUpdateEntry.get().getKey();
            if (maxUpdateCount > UPDATE_THRESHOLD) {
                System.out.println("ALARM: Table [" + table + "] has " + maxUpdateCount 
                    + " updates, exceeding threshold of " + UPDATE_THRESHOLD);
                // You could hook in additional code here to send notifications, etc.
            }
        }
    }

    /**
     * Find the table with the maximum DELETE count, trigger alarm if over threshold.
     */
    private static void checkAlarmForDeletes(List<Map.Entry<String, TableStats>> sortedEntries) {
        Optional<Map.Entry<String, TableStats>> maxDeleteEntry = sortedEntries.stream()
            .max(Comparator.comparingLong(e -> e.getValue().deleteCount));
        if (maxDeleteEntry.isPresent()) {
            long maxDeleteCount = maxDeleteEntry.get().getValue().deleteCount;
            String table = maxDeleteEntry.get().getKey();
            if (maxDeleteCount > DELETE_THRESHOLD) {
                System.out.println("ALARM: Table [" + table + "] has " + maxDeleteCount 
                    + " deletes, exceeding threshold of " + DELETE_THRESHOLD);
                // You could hook in additional code here to send notifications, etc.
            }
        }
    }

    /**
     * A small class to hold counts for INSERT, UPDATE, and DELETE.
     */
    private static class TableStats {
        long insertCount = 0;
        long updateCount = 0;
        long deleteCount = 0;

        long totalChanges() {
            return insertCount + updateCount + deleteCount;
        }
    }
}
