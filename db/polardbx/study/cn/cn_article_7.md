# Introduction

An Interpretation of PolarDB-X Source Codes (7): Life of Private Protocol Connection (CN)  
https://www.alibabacloud.com/blog/an-interpretation-of-polardb-x-source-codes-7-life-of-private-protocol-connection-cn_599458

# Private Protocol

Below is a detailed, structured summary of the article describing **GalaxySQL’s private protocol** as it operates on the **compute node (CN) side** in PolarDB-X. This summary covers the motivation behind introducing a custom protocol, the design of the network layer, how connections and sessions are managed, and how a simple query travels through this pipeline.

------

## 1. Motivation & Overview

1. **Why a private protocol?**
   - PolarDB-X’s computing and storage are disaggregated, with data physically residing on multiple shards (DNs).
   - A single logical SQL statement can fan out to many parallel sub-queries on each shard.
   - Traditional MySQL protocol + connection pooling can become a bottleneck at large scale.
   - **Solution**: A custom RPC-like protocol that decouples *connection* (TCP) from *session*, supports multiplexing, provides request pipelining, traffic control, and high-throughput data exchange.
2. **High-level concept**
   - The compute node (CN) acts as a **client** that sends pushdown queries (simple point lookups or complex joins) to the data nodes (DNs).
   - The private protocol ensures minimal overhead and maximum data locality by pushing down as much computation as possible to DNs.

------

## 2. Network Layer Framework

1. **Custom Reactor with Java NIO**
   - GalaxySQL’s network layer uses an in-house **Reactor** pattern (no external libraries).
   - A **NIOWorker** spawns multiple **NIOProcessor** instances (up to 2×CPU cores, capped at 32).
   - Each **NIOProcessor** wraps a **NIOReactor**, handling events (reads/writes) from the TCP sockets.
2. **Off-heap Buffers & I/O Flow**
   - Each Reactor maintains an off-heap memory pool:
     - Typically in 64KB chunks (for speed and reduced GC).
     - Large messages temporarily allocate on-heap buffers (then revert to off-heap when possible).
   - **Incoming data**: Arrives in off-heap buffers; Protobuf decoding places the parsed result in the heap.
   - **Outgoing data**: Batched into off-heap buffers. If the data is larger than a chunk, an on-heap buffer is used for serialization.
3. **Send & Receive Mechanics**
   - **Sending**: Requests are queued in a buffer, and a dedicated write thread pushes them into the socket’s send buffer.
   - **Receiving**: On a read event, the Reactor calls a callback to decode the received bytes into Protobuf messages.
   - **Flow control**: The private protocol can pipeline multiple sessions on one TCP channel, leveraging full-duplex capability.

------

## 3. Connection & Session

1. **Separation of TCP Connection vs. Session**
   - A **TCP connection** (wrapped in `XClient`) is long-lived and can carry many sessions in parallel.
   - A **session** (`XSession`) is the core context that associates each request with a state (e.g., transaction mode, default schema, etc.).
2. **`XClient`**
   - Represents the logical TCP connection:
     - Manages authentication (MySQL 4.1–style handshake).
     - Tracks all active sessions (`workingSessionMap`).
     - Handles lifecycle events (connect, disconnect, cleanup).
3. **`XSession`**
   - Binds to a session ID on the data node.
   - Stores session state: query plan, variables, transaction info, etc.
   - Receives asynchronous callbacks for result packets, error handling, and session-level flow control.

------

## 4. Connection Pool & Global Manager

1. **`XClientPool`**
   - Manages TCP connections and sessions for a **single** DN, identified by (IP, Port, Username).
   - Tracks free/idle sessions, open sessions, and concurrency limits.
   - Implements session “acquisition” (similar to `getConnection` in JDBC), pre-allocation, health checks, and idle release.
2. **`XConnectionManager`**
   - A **global** singleton that manages multiple `XClientPool`s—one pool for each unique DN triplet.
   - Provides scheduled tasks to:
     - Preheat connections.
     - Remove stale connections.
     - Enforce session/connection TTL (time-to-live).

------

## 5. JDBC Compatibility Layer

1. **Why JDBC-compatibility?**
   - To make the transition from JDBC to the new private protocol easier without heavy refactoring.
   - Lets developers switch protocols with minimal changes in the upper layer.
2. **Implementation**
   - Classes like `XConnection`, `XStatement`, `XPreparedStatement`, `XResultSet` implement or wrap JDBC interfaces.
   - Unsupported methods throw exceptions to prevent silent failures.
   - A “hot switching” mechanism can switch between MySQL protocol and the private protocol.

------

## 6. Lifecycle Walkthrough (From CN Perspective)

To illustrate the flow, the article shows a **“select 1”** example using `GalaxyTest`. Key steps:

1. **Data Source Initialization**

   - A new `XDataSource` is created for the target DN.
   - Internally, `XConnectionManager` is updated (registering or referencing an existing `XClientPool`).

2. **Obtain a Connection (`XConnection`)**

   - ```
     dataSource.getConnection()
     ```

     :

     - Finds the correct `XClientPool` based on (IP, Port, Username).
     - Tries to reuse an **idle** session if available; if not, it opens or reuses a TCP connection.
     - If needed, it creates a **new** `XClient` (TCP connect + authentication).
     - Creates a **new** `XSession` (assigns session ID, sets state to INIT).

3. **Send a Query**

   - ```
     conn.execQuery("select 1")
     ```

     :

     - Ensures the session is fully initialized (e.g., lazy creation if brand-new).
     - Constructs a Protobuf message with `SESS_NEW`, character set commands, the actual SQL request.
     - Writes the request into the off-heap send buffer (pipelined with any other pre-requests).

4. **Receive & Parse Results**

   - An `XResult` object is returned to handle incoming result packets in a streaming or non-streaming manner.

   - The 

     result-set state machine

      is triggered upon receiving packets from the DN:

     1. **Column metadata** (`RESULTSET_COLUMN_META_DATA`)
     2. **Rows** (`RESULTSET_ROW`)
     3. **Fetch done** (`RESULTSET_FETCH_DONE`)
     4. **Notices** (rows affected, warnings, etc.)
     5. **OK** packet (`SQL_STMT_EXECUTE_OK`) signifying request completion.

   - In “streaming mode,” each call to `result.next()` drives state machine consumption of one row at a time.

5. **Cleanup**

   - After execution:
     - The session can be returned to the pool as an idle session (if auto-commit) or remain in a transaction state.
     - The TCP connection (`XClient`) remains alive for further session reuse unless closed by eviction or TTL.

------

## 7. Advanced Features (Beyond the Example)

While the demo focuses on a single request (`select 1`), in practice the private protocol:

- Supports **pipelined** requests (multiple queries in-flight on the same session).
- Implements **traffic control** and chunk-based result streaming for large data sets.
- Optimizes **execution plan pushdown**, enabling partial or entire query plans to run on the DN side.

------

## 8. Conclusion

This article dives into **how a PolarDB-X compute node** uses a **private protocol** to interact with storage nodes. Key elements include:

- **NIO-based Reactor** for high-performance, asynchronous I/O.
- **Decoupling of TCP connections from sessions** for efficient multiplexing.
- A **connection pool** (`XClientPool`) and global manager (`XConnectionManager`) to handle concurrency, preheating, and resource lifecycle.
- A **JDBC compatibility layer** (`XConnection`, `XStatement`, etc.) for easy integration.
- A **state-machine** approach to parsing results, supporting both streaming and non-streaming reads.

Ultimately, the private protocol allows PolarDB-X to *push down more computation* and handle **many shards** concurrently while minimizing network overhead—crucial for large-scale, distributed SQL processing.