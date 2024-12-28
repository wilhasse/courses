import com.alibaba.polardbx.rpc.pool.XConnection;
import com.alibaba.polardbx.rpc.pool.XConnectionManager;
import com.alibaba.polardbx.rpc.result.XResult;
import com.alibaba.polardbx.rpc.result.XResultUtil;
import com.google.protobuf.ByteString;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.TimeZone;

public class QuerySQL {
    
    public static void main(String[] args) {
        try {
            // Get connection manager instance
            XConnectionManager manager = XConnectionManager.getInstance();
            
            // Basic connection details
            String host = "10.1.1.148";   // Changed from localhost to actual host
            int port = 33660;             // Changed from 3306 to X Protocol port
            String username = "teste";    // Changed from root
            String password = "teste";    // Changed from root
            String defaultDB = "mysql";   // Empty string for no default DB
            long timeoutNanos = 30000 * 1000000L; // Increased to 30 seconds         
           
            System.out.println("Initializing connection to " + host + ":" + port);
            
            // Initialize the datasource with longer timeout
            manager.initializeDataSource(host, port, username, password, "test-instance");
            
            try {
                // Get connection using the correct method with longer timeout
                System.out.println("Attempting to establish connection...");
                try (XConnection connection = manager.getConnection(host, port, username, password, defaultDB, timeoutNanos)) {
                    System.out.println("Connection established successfully!");
                    
                    // Enable stream mode
                    connection.setStreamMode(true);
                    
                    // After establishing connection, you should select database
                    connection.execUpdate("USE " + defaultDB);                    

                    try (Scanner scanner = new Scanner(System.in)) {
                        while (true) {
                            System.out.println("\nEnter SQL query (or 'exit' to quit):");
                            System.out.flush(); 
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
                               
                                // Print results
                                while (result.next() != null) {
                                    for (int i = 0; i < result.getMetaData().size(); i++) {
                                        Object value = XResultUtil.resultToObject(
                                            result.getMetaData().get(i),
                                            result.current().getRow().get(i),true,
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
                }
            } finally {
                // Cleanup
                System.out.println("Cleaning up connection...");
                manager.deinitializeDataSource(host, port, username, password);
            }
            
        } catch (Exception e) {
            System.err.println("Connection error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}