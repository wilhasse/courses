Below is a **step-by-step explanation** of the code that uses the **MySQL X DevAPI** to connect to a MySQL server, run an SQL statement (`SHOW DATABASES`), and print the results. This explanation aims to clarify both the **Java** details and the **MySQL X DevAPI** concepts involved.

------

## High-Level Overview

1. **Establish a Session** with a MySQL server using **X DevAPI** (connecting on port 33060, MySQL's X protocol port).
2. **Execute an SQL query** (`SHOW DATABASES`) via the X DevAPI interface.
3. **Fetch and display** the columns and rows returned by the query.
4. **Close** the session cleanly.

**Note**: The **MySQL X DevAPI** is different from the traditional MySQL protocol (which is typically on port 3306). MySQL X DevAPI (usually on port 33060) provides a more modern API for accessing MySQL, supporting both SQL and NoSQL (document store) operations.

------

## Code Breakdown

```java
import com.mysql.cj.xdevapi.*;
import java.util.List;

// Show All Databases
public class ListDBs {
    public static void main(String[] args) {
        // Connection parameters
        String host = "10.1.1.148";
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
```

### 1. Import Statements

```java
import com.mysql.cj.xdevapi.*;
import java.util.List;
```

- **`com.mysql.cj.xdevapi.\*`**: Provides the classes needed to use MySQL X DevAPI in Java.
- **`java.util.List`**: Used when we retrieve the list of column definitions.

------

### 2. Main Class and Method

```java
public class ListDBs {
    public static void main(String[] args) {
        // ...
    }
}
```

- Defines a standard Java `main` class to run the application from the command line.

------

### 3. Connection Parameters

```java
String host = "10.1.1.148";
int port = 33060;
String user = "teste";
String password = "teste";
```

- **`host`**: The IP address or hostname of the MySQL server.
- **`port`**: 33060 is the default port for the **MySQL X protocol** (as opposed to 3306 for the classic MySQL protocol).
- **`user`** and **`password`**: Credentials for connecting to the server.

------

### 4. Build the Connection URL

```java
String connectionUrl = String.format(
    "mysqlx://%s:%d?xdevapi.ssl-mode=DISABLED&user=%s&password=%s",
    host, port, user, password
);
System.out.printf("Url %s", connectionUrl);
System.out.println();
```

- **`mysqlx://...`**: X DevAPI connection protocol.
- **`?xdevapi.ssl-mode=DISABLED`**: Disables SSL for simplicity.
- **`user`** and **`password`**: Provided as query parameters.
- Prints out the constructed URL for debugging.

**Example**:

```
mysqlx://10.1.1.148:33060?xdevapi.ssl-mode=DISABLED&user=teste&password=teste
```

------

### 5. Create a Session

```java
Session session = new SessionFactory().getSession(connectionUrl);
```

- **`SessionFactory().getSession(...)`**: Creates a session (similar to a connection) to the MySQL server using the given URL.
- If successful, we’ll get a live session (`Session`) that we can use to run queries.

------

### 6. Execute a SQL Statement

```java
SqlStatement stmt = session.sql("SHOW DATABASES");
SqlResult result = stmt.execute();
```

- **`session.sql("SHOW DATABASES")`**: Prepares an SQL statement within the X DevAPI session.
- **`stmt.execute()`**: Executes the statement, returning a `SqlResult` object.

**`SHOW DATABASES`** is a standard SQL statement in MySQL that lists all available databases.

------

### 7. Retrieve the Columns from the Result

```java
List<Column> columns = result.getColumns();
for (Column col : columns) {
    System.out.printf("%-20s", col.getColumnName());
}
System.out.println();
```

- **`result.getColumns()`**: Fetches a list of `Column` objects representing the metadata of each column in the result set.
- The loop prints the column names in a 20-character wide format (`%-20s`).

------

### 8. Print a Separator

```java
for (int i = 0; i < columns.size(); i++) {
    System.out.print("--------------------");
}
System.out.println();
```

- Simply prints a line of dashes for each column, matching the header width.

------

### 9. Fetch and Print Rows

```java
Row row;
while ((row = result.fetchOne()) != null) {
    for (int i = 0; i < columns.size(); i++) {
        System.out.printf("%-20s", row.getString(i));
    }
    System.out.println();
}
```

- **`result.fetchOne()`**: Retrieves one row at a time from the result set (returns `null` when there are no more rows).
- **`row.getString(i)`**: Gets the row value for the **i**-th column as a string.
- Print each column in a fixed 20-character width to align the text nicely.

For `SHOW DATABASES`, a typical row might be something like `["information_schema"]`, `["mysql"]`, etc.

------

### 10. Close the Session

```java
} finally {
    session.close();
}
```

- Ensures that the `session` is **always closed** whether or not an exception occurred (by using a `finally` block).
- Properly releasing resources is important for database connections.

------

### 11. Exception Handling

```java
} catch (Exception e) {
    System.err.println("Error: " + e.getMessage());
    e.printStackTrace();
}
```

- If anything goes wrong (e.g., network error, invalid credentials, server unreachable), an exception is thrown.
- This block catches and prints the error message and stack trace.

------

## How It All Works Together

1. **The program** starts, sets up connection parameters for X DevAPI on `host:port`.

2. **Builds the connection URL** (including credentials and SSL options).

3. **Creates a `Session`** using `SessionFactory().getSession(...)`.

4. **Executes `SHOW DATABASES`** via the session’s `sql(...)` method.

5. **Retrieves column metadata** and **prints column names** as headers.

6. Fetches each row

    until there are no more:

   - Prints the database names in a neatly formatted manner.

7. **Closes the session** in a `finally` block to ensure the connection is released.

8. If any errors occur, prints an error message and stack trace.

------

## Key Points and Observations

- **Port 33060**: This is the **MySQL X Protocol** port, different from the standard 3306. Make sure the MySQL server is configured to listen on this port and that the X Plugin is enabled.
- **X DevAPI vs. JDBC**: This code uses the X DevAPI classes (e.g., `Session`, `SqlStatement`, `SqlResult`) from `com.mysql.cj.xdevapi.*` instead of the traditional JDBC API (`java.sql.Connection`, `java.sql.Statement`, etc.).
- **SSL Mode**: The URL query parameter `xdevapi.ssl-mode=DISABLED` explicitly disables SSL. In production, you’d typically use a secure mode.

By following the above steps, you can see how the code **connects to MySQL via X DevAPI**, **queries** the database, and **prints** the results all in a few lines of code.