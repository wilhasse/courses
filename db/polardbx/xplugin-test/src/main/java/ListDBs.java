import com.mysql.cj.xdevapi.*;
import java.util.List;

// Show All Databases
public class ListDBs {
    public static void main(String[] args) {
        // Connection parameters
        String host = "10.1.1.158";
        int port = 33060;
        String user = "teste";
        String password = "teste";
        
        try {
            // Create client session with SSL disabled
            String connectionUrl = String.format(
                "mysqlx://%s:%d?xdevapi.ssl-mode=DISABLED&user=%s&password=%s",
                host, port, user, password
            );
            System.out.printf("Url %s", connectionUrl);
            System.out.println();
            
            Session session = new SessionFactory().getSession(connectionUrl);
            
            try {
                System.out.println("Connected successfully!");
                
                // Execute a simple SQL query
                SqlStatement stmt = session.sql("SHOW DATABASES");
                SqlResult result = stmt.execute();
                
                // Print headers
                List<Column> columns = result.getColumns();
                for (Column col : columns) {
                    System.out.printf("%-20s", col.getColumnName());
                }
                System.out.println();
                
                // Print separator
                for (int i = 0; i < columns.size(); i++) {
                    System.out.print("--------------------");
                }
                System.out.println();
                
                // Print rows
                Row row;
                while ((row = result.fetchOne()) != null) {
                    for (int i = 0; i < columns.size(); i++) {
                        System.out.printf("%-20s", row.getString(i));
                    }
                    System.out.println();
                }
                
            } finally {
                session.close();
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}