# Introduction

PolarDB-X Plugin Server Test

# Build

```bash
mvn clean package
```

Check dependency Tree

```bash
mvn dependency:tree
```

Run the server:

```bash
java -jar target\servertest-1.0-SNAPSHOT.jar
Starting Server on port 8507
Processor 0 started.
Processor 1 started.
Processor 2 started.
Processor 3 started.
Processor 4 started.
Processor 5 started.
Processor 6 started.
Processor 7 started.
Server fully started on port 8507
Server started successfully. Press Ctrl+C to stop.
```

# Classes

Below is a very concise explanation of each class in the code:

1. **MultiServerConnectionPool**  
   Manages a collection of backend servers (each with a pool of connections) and provides round-robin access to valid connections. Automatically replaces connections if they become invalid.

2. **MyQueryHandler**  
   Receives SQL queries from the client, determines if they can be parallel-chunked, executes those chunks across multiple connections, merges the results, and sends them back to the client.

3. **Server**  
   A simple standalone server that sets up the polardbx-net NIO infrastructure (processors, acceptor) and uses a custom connection factory to create `ServerConnection` objects.

4. **ServerConnection**  
   Extends the polardbx-net `FrontendConnection`, handling authentication, buffer allocation, and delegating queries to `MyQueryHandler`.

5. **ServerInfo**  
   A plain data holder (POJO) that contains the host, port, username, password, and default database details for a backend server.

6. **ChunkResult**  
   Stores the outcome of a single chunk query execution: includes the `XResult` metadata and the fully retrieved rows.

7. **ChunkIterator**  
   Provides an iterator-like mechanism for merging sorted rows from each chunk by exposing the current row and advancing through them.

8. **TimestampLogger**  
   Offers simple timestamp-based logging utilities for measuring and reporting how long certain operations (like chunk queries) take.
