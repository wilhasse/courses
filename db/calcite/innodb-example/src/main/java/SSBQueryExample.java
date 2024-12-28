import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Properties;

public class SSBQueryExample {
    public static void main(String[] args) throws Exception {
        // Load the Calcite JDBC driver
        Class.forName("org.apache.calcite.jdbc.Driver");

        // Connection properties
        Properties info = new Properties();
        info.setProperty("model", "src/main/resources/ssb-model.json");
        
        // Create connection
        try (Connection connection = DriverManager.getConnection("jdbc:calcite:", info);
             Statement statement = connection.createStatement()) {
            
            // Array of queries to execute
            String[] queries = new String[] {
                // Q1.1
                "SELECT sum(\"lineorder\".\"lo_extendedprice\" * \"lineorder\".\"lo_discount\") AS \"revenue\"\n" +
                "FROM \"lineorder\", \"date\"\n" +
                "WHERE \"lineorder\".\"lo_orderdate\" = \"date\".\"d_datekey\"\n" +
                "AND \"date\".\"d_year\" = 1993\n" +
                "AND \"lineorder\".\"lo_discount\" between 1 and 3\n" +
                "AND \"lineorder\".\"lo_quantity\" < 25",
                
                // Table counts
                "SELECT COUNT(*) AS \"count\" FROM \"lineorder\"",
                "SELECT COUNT(*) AS \"count\" FROM \"customer\"",
                "SELECT COUNT(*) AS \"count\" FROM \"supplier\"",
                "SELECT COUNT(*) AS \"count\" FROM \"part\"",
                "SELECT COUNT(*) AS \"count\" FROM \"date\""
            };
            
            // Execute each query
            for (String query : queries) {
                System.out.println("\nExecuting query:\n" + query);
                try (ResultSet resultSet = statement.executeQuery(query)) {
                    // Print all columns from the result
                    while (resultSet.next()) {
                        for (int i = 1; i <= resultSet.getMetaData().getColumnCount(); i++) {
                            String columnName = resultSet.getMetaData().getColumnLabel(i);
                            Object value = resultSet.getObject(i);
                            System.out.printf("%s: %s\n", columnName, value);
                        }
                    }
                }
            }
        }
    }
}