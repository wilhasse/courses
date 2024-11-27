import com.alibaba.polardbx.rpc.client.XClient;
import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.pool.XConnectionManager;
import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.common.jdbc.BytesSql;
import com.alibaba.polardbx.rpc.result.XResultUtil;
import com.google.protobuf.ByteString;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.TimeZone;
import java.util.concurrent.atomic.AtomicLong;

public class SimpleDbQueryApp {
    private static final AtomicLong sessionIdGenerator = new AtomicLong(1);
    
    public static void main(String[] args) {
        try {
            // Get connection manager instance
            XConnectionManager manager = XConnectionManager.getInstance();
            
            // Basic connection details
            String host = "10.1.1.158";
            int port = 33060;
            String username = "teste";
            String password = "teste";
            String defaultDB = "test"; // Default database name
            long timeoutNanos = 5000 * 1000000L; // 5 seconds in nanos
            
            // Initialize the datasource
            manager.initializeDataSource(host, port, username, password, "test-instance");
            
            // Get connection using the correct method
            try (XConnection connection = manager.getConnection(host, port, username, password, defaultDB, timeoutNanos)) {
                Scanner scanner = new Scanner(System.in);
                while (true) {
                    System.out.println("\nEnter SQL query (or 'exit' to quit):");
                    String sqlInput = scanner.nextLine();
                    
                    if ("exit".equalsIgnoreCase(sqlInput)) {
                        break;
                    }
                    
                    try {
                        // Execute query using the raw SQL string
                        XResult result = connection.execQuery(sqlInput);
                        
                        // Print column headers
                        List<String> columns = new ArrayList<>();
                        for (int i = 0; i < result.getMetaData().size(); i++) {
                            ByteString colName = result.getMetaData().get(i).getName();
                            columns.add(colName.toStringUtf8());
                            System.out.print(String.format("%-20s", colName.toStringUtf8()));
                        }
                        System.out.println();
                        
                        // Print separator
                        for (String ignored : columns) {
                            System.out.print("--------------------");
                        }
                        System.out.println();
                        
                        // Print results
                        while (result.next() != null) {
                            for (int i = 0; i < result.getMetaData().size(); i++) {
                                Object value = XResultUtil.resultToObject(
                                    result.getMetaData().get(i),
                                    result.current().getRow().get(i),
                                    true,
                                    TimeZone.getDefault()
                                ).getKey();
                                System.out.print(String.format("%-20s", value != null ? value.toString() : "NULL"));
                            }
                            System.out.println();
                        }
                    } catch (Exception e) {
                        System.err.println("Error executing query: " + e.getMessage());
                        e.printStackTrace();
                    }
                }
            }
            
            // Cleanup
            manager.deinitializeDataSource(host, port, username, password);
            
        } catch (Exception e) {
            System.err.println("Connection error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}