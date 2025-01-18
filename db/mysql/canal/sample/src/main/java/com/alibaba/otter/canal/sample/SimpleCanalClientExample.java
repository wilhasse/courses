package com.alibaba.otter.canal.sample;

import java.net.InetSocketAddress;
import java.util.List;

import com.alibaba.otter.canal.client.CanalConnectors;
import com.alibaba.otter.canal.client.CanalConnector;
import com.alibaba.otter.canal.protocol.Message;
import com.alibaba.otter.canal.protocol.CanalEntry.Column;
import com.alibaba.otter.canal.protocol.CanalEntry.Entry;
import com.alibaba.otter.canal.protocol.CanalEntry.EntryType;
import com.alibaba.otter.canal.protocol.CanalEntry.EventType;
import com.alibaba.otter.canal.protocol.CanalEntry.RowChange;
import com.alibaba.otter.canal.protocol.CanalEntry.RowData;

/**
 * A simple Canal client example.
 * 
 * This client demonstrates how to:
 * 1. Connect to a Canal server.
 * 2. Subscribe to binlog changes.
 * 3. Fetch new messages (events) from the server.
 * 4. Process and print out the row changes (INSERT, UPDATE, DELETE).
 */
public class SimpleCanalClientExample {

    public static void main(String[] args) {

        /**
         * Create a CanalConnector to connect to the Canal server.
         * newSingleConnector(...) is used to connect to a single Canal instance.
         *
         * Parameters:
         * - InetSocketAddress: the address of the Canal server (host and port).
         * - destination: the name of the destination instance (here it's "example").
         * - username and password: the credentials if Canal server requires authentication (left empty here).
         */
        CanalConnector connector = CanalConnectors.newSingleConnector(
            new InetSocketAddress("10.200.15.8", 11111), 
            "example", 
            "", 
            ""
        );

        // The number of entries fetched per request.
        int batchSize = 1000;

        // Used to count how many times we've received an empty message batch.
        int emptyCount = 0;

        try {
            // Connect to the Canal server.
            connector.connect();

            // Subscribe to all changes in all schemas: ".*\\..*"
            // This means you want to capture all events from all databases and tables.
            connector.subscribe(".*\\..*");

            // Rollback any previous incomplete acknowledgments to start fresh.
            connector.rollback();

            // The max number of empty messages to receive before exiting.
            int totalEmptyCount = 120;

            // Continuously attempt to fetch new binlog events until we have too many empty results.
            while (emptyCount < totalEmptyCount) {

                // Fetch messages without automatically acknowledging them (i.e., commit).
                Message message = connector.getWithoutAck(batchSize);

                // Each Message has a batchId and a list of Entries.
                long batchId = message.getId();
                int size = message.getEntries().size();

                if (batchId == -1 || size == 0) {
                    // Received an empty message, increment the emptyCount.
                    emptyCount++;
                    System.out.println("empty count : " + emptyCount);

                    // Sleep for 1 second and then continue fetching.
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException e) {
                        // Handle interruption.
                    }
                } else {
                    // If we did get entries, reset emptyCount.
                    emptyCount = 0;
                    printEntry(message.getEntries());
                }

                /**
                 * connector.ack(batchId) acknowledges (commits) that we've successfully processed
                 * all entries in this batch. If you encounter an error while processing, you would
                 * call connector.rollback(batchId) instead to signal the server that you haven't
                 * fully consumed/processed the entries.
                 */
                connector.ack(batchId);
            }

            System.out.println("empty too many times, exit");
        } finally {
            // Always disconnect the connector at the end to free resources.
            connector.disconnect();
        }
    }

    /**
     * Iterates over each Entry in the list and processes each row change.
     * 
     * @param entries List of binlog entries.
     */
    private static void printEntry(List<Entry> entries) {
        for (Entry entry : entries) {
            // We skip transaction BEGIN and END events, and only focus on row-level events.
            if (entry.getEntryType() == EntryType.TRANSACTIONBEGIN
                    || entry.getEntryType() == EntryType.TRANSACTIONEND) {
                continue;
            }

            RowChange rowChange;
            try {
                // Convert the raw storeValue into a RowChange object which contains row-level data.
                rowChange = RowChange.parseFrom(entry.getStoreValue());
            } catch (Exception e) {
                throw new RuntimeException(
                    "ERROR parsing event data: " + entry.toString(), e
                );
            }

            EventType eventType = rowChange.getEventType();
            System.out.println(String.format(
                "================> binlog[%s:%s] , name[%s,%s] , eventType : %s",
                entry.getHeader().getLogfileName(),
                entry.getHeader().getLogfileOffset(),
                entry.getHeader().getSchemaName(),
                entry.getHeader().getTableName(),
                eventType
            ));

            // Each RowChange can contain multiple RowData objects.
            for (RowData rowData : rowChange.getRowDatasList()) {
                if (eventType == EventType.DELETE) {
                    // For DELETE, the data is in "beforeColumnsList".
                    printColumn(rowData.getBeforeColumnsList());
                } else if (eventType == EventType.INSERT) {
                    // For INSERT, the data is in "afterColumnsList".
                    printColumn(rowData.getAfterColumnsList());
                } else {
                    // For UPDATE, we can see both "before" and "after" states of each column.
                    System.out.println("-------> before");
                    printColumn(rowData.getBeforeColumnsList());
                    System.out.println("-------> after");
                    printColumn(rowData.getAfterColumnsList());
                }
            }
        }
    }

    /**
     * Prints information for each column in a row.
     *
     * @param columns List of Column objects to be printed.
     */
    private static void printColumn(List<Column> columns) {
        for (Column column : columns) {
            // column.getUpdated() indicates whether the column was updated in this event.
            System.out.println(
                column.getName() + " : " + column.getValue() + "    update=" + column.getUpdated()
            );
        }
    }
}
