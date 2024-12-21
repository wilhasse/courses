# SimpleServer.java

Below is a step-by-step walkthrough and explanation of what the code is doing. Essentially, it implements a rudimentary MySQL-like server that listens for client connections on a specified port (8507), uses an internal query handler that connects to a PolarDB-X backend, executes SQL queries there, and then returns results back to the client using the MySQL protocol.

------

## Overview

- **SimpleServer** is a singleton class that starts up an NIO-based server listening on a port (8507).
- It uses a custom **DebugConnection** as the connection type for each client.
- It uses **SimplePrivileges** for user authentication (very simple user/pass checks from a map).
- It uses **SimpleQueryHandler** to handle each SQL query by forwarding them to PolarDB-X.
- It then writes back the results in MySQL wire protocol format.

Once started:

1. A timer updates the `TimeUtil` periodically (used in some MySQL protocol contexts).
2. A number of **NIOProcessor** threads are created for handling connections I/O.
3. A **NIOAcceptor** is created and started to listen for new connections on port 8507.
4. When a new connection arrives, `DebugConnectionFactory` instantiates a **DebugConnection** object.
5. Within the DebugConnection, a **SimpleQueryHandler** is set. This handler opens a backend connection to PolarDB-X (via `XConnectionManager`), sends queries there, and streams results back to the MySQL client.

------

## Detailed Walkthrough

### 1. Main Entry Point

```
public static void main(String[] args) {
    ...
    getInstance().startup();
    ...
}
```

- The `main` method is the starting point. It calls `SimpleServer.getInstance().startup()`.
- After a successful startup, it goes into an infinite loop, effectively keeping the server running.

### 2. The `SimpleServer` Singleton

```
private static final SimpleServer INSTANCE = new SimpleServer();

public static SimpleServer getInstance() {
    return INSTANCE;
}
```

- The server uses a singleton pattern to ensure only one instance is created.

### 3. Startup Process

```
public void startup() throws IOException {
    System.out.println("Starting server initialization...");
    ...
    // 1) Timer scheduling
    // 2) NIOProcessor creation and startup
    // 3) NIOAcceptor creation and startup
}
```

1. **Timer Initialization**

   ```
   Timer timer = new Timer("ServerTimer", true);
   timer.schedule(new TimerTask() {
       @Override
       public void run() {
           TimeUtil.update();
       }
   }, 0L, 100L);
   ```

   - Creates a background timer that calls `TimeUtil.update()` every 100ms.
   - This is often used by MySQL-based servers to keep track of the current time more efficiently rather than calling `System.currentTimeMillis()` repeatedly.

2. **NIOProcessor Creation**

   ```
   int processorCount = Math.max(1, ThreadCpuStatUtil.NUM_CORES);
   processors = new NIOProcessor[processorCount];
   for (int i = 0; i < processors.length; i++) {
       ServerThreadPool handler = new ServerThreadPool(
           "ProcessorHandler-" + i,
           4,  // poolSize
           5000,  // deadLockCheckPeriod (5 seconds)
           1   // bucketSize
       );
       processors[i] = new NIOProcessor(i, "Processor" + i, handler);
       processors[i].startup();
   }
   ```

   - Calculates the number of processors to use (based on CPU cores).
   - For each processor, creates a **ServerThreadPool** and instantiates an **NIOProcessor**.
   - The **NIOProcessor** is responsible for handling network I/O events in a non-blocking fashion.
   - Calls `startup()` on each processor to start its internal threads.

3. **NIOAcceptor** Creation (the actual server)

   ```
   DebugConnectionFactory factory = new DebugConnectionFactory();
   server = new NIOAcceptor("MySQLServer", SERVER_PORT, factory, true);
   server.setProcessors(processors);
   server.start();
   ```

   - Creates a **NIOAcceptor**, which is the actual server listening for incoming connections on port 8507.
   - `DebugConnectionFactory` tells the server how to construct a new connection object (our custom `DebugConnection`) for each incoming socket.
   - Calls `server.start()` to begin listening for connections.

### 4. `SimpleConfig` (Holds Users)

```
static class SimpleConfig {
    private final Map<String, String> users;

    public SimpleConfig() {
        this.users = new HashMap<>();
        this.users.put("root", "12345");
    }

    public Map<String, String> getUsers() {
        return users;
    }
}
```

- A very basic config: stores a map of user->password.
- By default, it has just one user: `root` with password `12345`.

### 5. `DebugConnectionFactory`

```
class DebugConnectionFactory extends FrontendConnectionFactory {
    @Override
    protected FrontendConnection getConnection(SocketChannel channel) {
        ...
        DebugConnection c = new DebugConnection(channel);
        c.setPrivileges(new SimplePrivileges());
        c.setQueryHandler(new SimpleQueryHandler(c));
        return c;
    }
}
```

- Whenever a new client connects, the server will invoke `getConnection(...)`.
- It creates a `DebugConnection`, sets `SimplePrivileges`, and the `SimpleQueryHandler`.
- Returns it to the acceptor so it can start handling I/O for that connection.

### 6. `DebugConnection`

```
class DebugConnection extends FrontendConnection {
    private final BufferPool bufferPool;
    private final AtomicLong CONNECTION_ID = new AtomicLong(1);
    private final long connectionId;
    private final SimpleQueryHandler queryHandler;
    ...
}
```

- **DebugConnection** extends `FrontendConnection`, which is a custom connection class that handles MySQL wire protocol with non-blocking I/O.
- Allocates a **BufferPool** (16 MB overall, 4 KB chunks) to manage ByteBuffers efficiently.
- Maintains an atomic connection ID (so each connection has a unique ID).
- Has a **SimpleQueryHandler** for SQL queries.

#### `DebugConnection` constructor

```
public DebugConnection(SocketChannel channel) {
    super(channel);
    this.bufferPool = new BufferPool(1024 * 1024 * 16, 4096);
    ...
    this.queryHandler = new SimpleQueryHandler(this);
    this.connectionId = CONNECTION_ID.getAndIncrement();
    ...
}
```

- Initializes the buffer pool, sets packet limits, etc.
- Prints out debug logs.

#### `allocate()` and `recycleBuffer(...)`

- Manages ByteBuffers for reading/writing to the socket channel.
- Use the buffer pool to get a fresh ByteBuffer, and recycle when done.

#### `cleanup()`

- Cleans up when the connection is closed.
- (Commented out code to close the query handler, but in a real scenario you might close the database connection here.)

### 7. `SimplePrivileges`

```
class SimplePrivileges implements Privileges {
    @Override
    public boolean userExists(String user) {
        return getConfig().getUsers().containsKey(user);
    }
    ...
    @Override
    public EncrptPassword getPassword(String user) {
        String pass = getConfig().getUsers().get(user);
        return new EncrptPassword(pass, false);
    }
    ...
}
```

- A bare-bones privileges implementation that checks if a user is in the `SimpleConfig.users` map.
- Returns the password from that map.
- Ignores other details like schemas, host, table privileges, etc.
- Always returns `true` for `isTrustedIp(...)`, so no real IP-based access control.

### 8. `SimpleQueryHandler`

```
class SimpleQueryHandler implements QueryHandler {
    private final DebugConnection connection;
    private final XConnection polardbConnection;
    private final XConnectionManager manager;
    ...
}
```

- This is where the actual SQL queries from the client are handled.
- On initialization, it:
  1. Fetches an `XConnectionManager` instance (singleton).
  2. Connects to PolarDB-X using `XConnectionManager.getConnection(...)`.
  3. Executes a `USE ssb` statement to set the default DB.

**Important**: The code example is connecting to a PolarDB-X instance at `10.1.1.148:33660` with username/password `teste` and default database `ssb`. This is just an example—replace with your actual coordinates if needed.

#### `query(String sql)`

```
@Override
public void query(String sql) {
    System.out.println("Received query: " + sql);
    try {
        XResult result = polardbConnection.execQuery(sql);
        sendResultSetResponse(result);
    } catch (Exception e) {
        sendErrorResponse(e.getMessage());
    }
}
```

- Receives a SQL query from the MySQL client (the one that connected to our server).
- Forwards the query to the PolarDB-X backend by calling `polardbConnection.execQuery(sql)`.
- On success, calls `sendResultSetResponse(...)`.
- On failure, sends an error packet back to the client.

#### `sendResultSetResponse(XResult result)`

```
private void sendResultSetResponse(XResult result) {
    // 1) Allocate a buffer
    // 2) Write a ResultSetHeaderPacket
    // 3) Write FieldPackets (one per column)
    // 4) Write an EOF (if not using deprecation mode)
    // 5) Fetch each row from XResult, build a RowDataPacket, and write it
    // 6) Write a final EOF
    ...
}
```

1. **Allocate a ByteBuffer** from the connection’s buffer pool.
2. **Write a MySQL result set header** (how many columns are returned).
3. **Write a `FieldPacket`** for each column in the result’s metadata, including the column’s name, type, etc.
4. **Write an `EOFPacket`** to indicate the end of the column metadata.
5. **Iterate over each row** in the `XResult`, converting the data to strings (or `null`). Each row is sent as a `RowDataPacket`.
6. **Write a final EOF** to mark the end of data.
7. The client reading this will see a well-formed MySQL result set.

Note: `convertPolarDBTypeToMySQLType(...)` is a helper that maps PolarDB-X’s column metadata to MySQL protocol type codes.

#### `sendErrorResponse(String message)`

- Sends a MySQL `ErrorPacket` to the client with an error code (1064) and an error message.
- The client will likely show this as a MySQL error.

#### `close()`

```
public void close() {
    ...
    polardbConnection.close();
    manager.deinitializeDataSource("10.1.1.148", 33660, "teste", "teste");
}
```

- Closes the `XConnection` to the PolarDB-X engine.
- Deinitializes the data source in the `XConnectionManager`.

------

## Putting It All Together

1. **Client connects** to `localhost:8507` (or wherever the server is listening).
2. `NIOAcceptor` accepts the connection, calls `DebugConnectionFactory` to create a `DebugConnection`.
3. `DebugConnection` does a handshake, using `SimplePrivileges` to authenticate the user (for example, user=`root`, password=`12345`).
4. The client sends a **SQL query** (e.g., `SELECT 1`).
5. `DebugConnection` delegates to `SimpleQueryHandler.query(...)`.
6. `SimpleQueryHandler` calls the PolarDB-X backend (`XConnection.execQuery(sql)`), retrieves the rows, and writes them back to the client in MySQL wire protocol.
7. Client sees normal MySQL result sets.

It is basically a **simple MySQL proxy** that:

- Listens on port 8507.
- Forwards queries to PolarDB-X (running at 10.1.1.148:33660).
- Writes the response back using MySQL’s standard protocol packets.

------

## Key Points

- This code is **not a complete production MySQL proxy** (many features are absent or simplified).
- It demonstrates how to:
  1. Use **NIO** (Non-blocking I/O) for server acceptor and connection handling.
  2. Implement a **MySQL wire protocol** handshake, authentication, and query flow.
  3. Integrate with a **PolarDB-X** backend using the `XConnectionManager`.
  4. Allocate and recycle buffers via a **BufferPool**.
- The crucial detail is converting the data from PolarDB-X (`XResult`) into MySQL packets (`FieldPacket`, `RowDataPacket`, `EOFPacket`, `ErrorPacket`, etc.).

------

### Summary

**In short, this code spins up a custom MySQL-like server on port 8507 that accepts incoming connections, authenticates them, and forwards their queries to a PolarDB-X instance. Then it sends back the results (or errors) in the MySQL protocol so that a standard MySQL client can interact with it.**