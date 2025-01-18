Below is a **high-level** explanation of the Canal protocol as defined by its Protocol Buffers (“.proto”) files. The protocol is split into three `.proto` files that handle **(1)** the main client/server communication handshake and messages, **(2)** the binlog row-data changes, and **(3)** administrative commands.

---

# 1. `CanalPacket.proto`

This file describes the **primary** communication packet structure and message types between a Canal client and the Canal server. It defines:

1. **Compression Enum** (`Compression`)
2. **PacketType Enum** (`PacketType`)
3. **Packet** (the **top-level** message wrapper)
4. **Handshake**, **ClientAuth**, **Ack**, **ClientAck** and other messages relevant to the standard client interaction flow (subscription, unsubscription, pulling data, etc.)

```protobuf
syntax = "proto3";
package com.alibaba.otter.canal.protocol;

option java_package = "com.alibaba.otter.canal.protocol";
option java_outer_classname = "CanalPacket";
option optimize_for = SPEED;

/** Indicates how the Canal server may compress the data in a Packet's `body`. */
enum Compression {
    COMPRESSIONCOMPATIBLEPROTO2 = 0; // legacy placeholder
    NONE = 1;                       // no compression
    ZLIB = 2;                       // zlib compression
    GZIP = 3;                       // gzip compression
    LZF = 4;                        // LZF compression
}

/**
 * Distinguishes different types of packets in the Canal communication protocol.
 * The client and server use PacketType to know how to interpret the packet's body.
 */
enum PacketType {
    PACKAGETYPECOMPATIBLEPROTO2 = 0; // legacy placeholder
    HANDSHAKE = 1;                  // server -> client: initial handshake
    CLIENTAUTHENTICATION = 2;       // client -> server: authentication info
    ACK = 3;                        // server -> client: or client -> server to confirm success
    SUBSCRIPTION = 4;               // client -> server: subscription request
    UNSUBSCRIPTION = 5;             // client -> server: unsubscription request
    GET = 6;                        // client -> server: fetch next batch of data
    MESSAGES = 7;                   // server -> client: the actual data messages
    CLIENTACK = 8;                  // client -> server: acknowledgment for data batch
    SHUTDOWN = 9;                   // management: request to shut down canal server
    DUMP = 10;                      // instruct server to dump binlog from certain position
    HEARTBEAT = 11;                 // heartbeat checking
    CLIENTROLLBACK = 12;            // client -> server: rollback a batch
}

/**
 * The main wrapper for all network packets in the standard Canal client-server protocol.
 * Every packet has:
 * - A 'magic_number' (legacy usage)
 * - A 'version'
 * - A 'type' (PacketType)
 * - An optional 'compression'
 * - A 'body' (the actual payload, which may be another protobuf-encoded message)
 */
message Packet {
    oneof magic_number_present {
        int32 magic_number = 1;
    }
    oneof version_present {
        int32 version = 2;
    }
    PacketType type = 3;
    oneof compression_present {
        Compression compression = 4;
    }
    bytes body = 5; // The payload (could be Handshake, ClientAuth, etc. in serialized form)
}

/** Simple heartbeat message to measure round-trip or detect idle connections. */
message HeartBeat {
    int64 send_timestamp = 1;
    int64 start_timestamp = 2;
}

/**
 * Sent by the server to the client upon initial connection.
 * `seeds` is used for salted password hashing.
 */
message Handshake {
    oneof communication_encoding_present {
        string communication_encoding = 1; // e.g. "utf8"
    }
    bytes seeds = 2; // random seed bytes
    Compression supported_compressions = 3; // what compression the server can handle
}

/**
 * The client’s response to a Handshake, providing authentication info.
 * - username, password: typical credentials
 * - net_read_timeout, net_write_timeout: optional timeouts
 * - destination: which Canal instance (pipeline) the client wants to subscribe to
 * - client_id: client identifier
 * - filter: table filter
 * - start_timestamp: used for resuming from a specific time (optional)
 */
message ClientAuth {
    string username = 1;
    bytes password = 2; 
    oneof net_read_timeout_present {
         int32 net_read_timeout = 3;
    }
    oneof net_write_timeout_present {
        int32 net_write_timeout = 4;
    }
    string destination = 5;
    string client_id = 6;
    string filter = 7;
    int64 start_timestamp = 8;
}

/**
 * Acknowledgment message. Could be sent from server -> client to indicate success,
 * or from client -> server if needed. Contains possible error_code/message.
 */
message Ack {
    oneof error_code_present {
        int32 error_code = 1;
    }
    string error_message = 2;
}

/**
 * A specialized acknowledgment from client -> server that includes:
 * - destination (which pipeline)
 * - client_id
 * - batch_id: identifying which batch of messages the client is ACKing
 */
message ClientAck {
    string destination = 1;
    string client_id = 2;
    int64 batch_id = 3;
}

/**
 * The client’s subscription request.
 * - destination: canal pipeline name
 * - client_id: client identifier
 * - filter: e.g., "db1\\..*" or ".*\\..*"
 */
message Sub {
    string destination = 1;
    string client_id = 2;
    string filter = 7;
}

/** The client’s request to unsubscribe. */
message Unsub {
    string destination = 1;
    string client_id = 2;
    string filter = 7;
}

/**
 * The client’s request to pull data from the server (polling).
 * - destination, client_id
 * - fetch_size: how many entries to fetch at once
 * - timeout, unit: optional waiting period
 * - auto_ack: if true, the server automatically considers this batch as acknowledged
 */
message Get {
    string destination = 1;
    string client_id = 2;
    int32 fetch_size = 3;
    oneof timeout_present {
        int64 timeout = 4; 
    }
    oneof unit_present {
        int32 unit = 5; 
    }
    oneof auto_ack_present {
        bool auto_ack = 6;
    }
}

/**
 * The response from the server to the client’s GET request.
 * - batch_id: identifies this batch of events
 * - messages: repeated bytes; each item is typically a serialized `Entry` or another structure
 */
message Messages {
    int64 batch_id = 1;
    repeated bytes messages = 2; 
}

/**
 * Request the server to dump binlog starting from a certain position or timestamp
 * - `journal` = binlog filename
 * - `position` = offset in that binlog
 * - `timestamp` can also be used
 */
message Dump {
    string journal = 1;
    int64  position = 2;
    oneof timestamp_present {
        int64 timestamp = 3;
    }
}

/**
 * Client -> server request to rollback a previously retrieved batch.
 * - This indicates the client could *not* process a batch successfully
 *   and wants the server to resend it.
 */
message ClientRollback {
    string destination = 1;
    string client_id = 2;
    int64 batch_id = 3;
}
```

### **Key Points about `CanalPacket.proto`**:
- **`Packet`** is the **top-level** container used for each request/response.  
- The `type` field (`PacketType`) tells you which *kind* of message is in the `body`.  
- The `body` is serialized data (e.g., `Handshake`, `ClientAuth`, `Messages`, etc.).  
- **Flow**:  
  1. Server sends `Packet(type=HANDSHAKE, body=Handshake)`.  
  2. Client replies with `Packet(type=CLIENTAUTHENTICATION, body=ClientAuth)`.  
  3. Server responds with `Packet(type=ACK, body=Ack)` to confirm success or an error.  
  4. Then the client may send `SUBSCRIPTION`, `GET` (pulling data), `CLIENTACK`, etc.  

---

# 2. `CanalEntry.proto`

This file describes the **internal data structure** used to represent row changes (the actual binlog events) once they are parsed. These messages are typically found in the `Messages.messages` field from the server. The key message is `Entry`, which contains a `Header` and a `storeValue` (often a serialized `RowChange`).

1. **`Entry`**  
2. **`Header`**  
3. **`Column`, `RowData`, `RowChange`**  
4. **TransactionBegin**, **TransactionEnd** (for capturing TX boundaries)  
5. **Pair** (a simple key-value extension)  
6. **`EntryType`** and **`EventType`** enums  
7. **`Type`** enum for the source DB engine

```protobuf
syntax = "proto3";
package com.alibaba.otter.canal.protocol;

option java_package = "com.alibaba.otter.canal.protocol";
option java_outer_classname = "CanalEntry";
option optimize_for = SPEED;

message Entry {
    Header header = 1;
    oneof entryType_present {
        EntryType entryType = 2; 
    }
    bytes storeValue = 3; // Typically a serialized RowChange, TransactionBegin, or TransactionEnd
}

message Header {
    oneof version_present {
        int32 version = 1; 
    }
    string logfileName = 2;   // binlog or redo log filename
    int64 logfileOffset = 3;  // offset in the log file
    int64 serverId = 4;       // originating DB server ID
    string serverenCode = 5;  // encoding of the DB server
    int64 executeTime = 6;    // when the event was executed
    oneof sourceType_present {
        Type sourceType = 7;  // e.g. MYSQL or ORACLE
    }
    string schemaName = 8;    // database/schema
    string tableName = 9;     // table
    int64 eventLength = 10;   // event body length
    oneof eventType_present {
        EventType eventType = 11; // e.g. INSERT, UPDATE, DELETE, etc.
    }
    repeated Pair props = 12; // key-value properties
    string gtid = 13;         // optional GTID if available
}

message Column {
    int32 index = 1;       // column index in the table
    int32 sqlType = 2;     // Java SQL Type (e.g. Types.VARCHAR = 12, Types.INTEGER = 4, etc.)
    string name = 3;       // column name
    bool isKey = 4;        // whether this column is part of the primary key
    bool updated = 5;      // for UPDATE: indicates if the column changed
    oneof isNull_present {
        bool isNull = 6;
    }
    repeated Pair props = 7; // extra metadata
    string value = 8;        // the actual column value (as text)
    int32 length = 9;        // original data length
    string mysqlType = 10;   // MySQL-specific type (e.g. "varchar(128)")
}

message RowData {
    repeated Column beforeColumns = 1; // columns before update/delete
    repeated Column afterColumns = 2;  // columns after update/insert
    repeated Pair props = 3;
}

message RowChange {
    int64 tableId = 1;
    oneof eventType_present {
        EventType eventType = 2; // INSERT, UPDATE, DELETE, DDL ops, etc.
    }
    oneof isDdl_present {
        bool isDdl = 10; // true if this row change is actually a DDL statement
    }
    string sql = 11;                // if isDdl = true, the actual DDL SQL
    repeated RowData rowDatas = 12; // the actual row changes
    repeated Pair props = 13; 
    string ddlSchemaName = 14;      // schema name under which this DDL is executed
}

message TransactionBegin {
    // (executeTime and transactionId are deprecated)
    int64 executeTime = 1;
    string transactionId = 2;
    repeated Pair props = 3;
    int64 threadId = 4; // the transaction’s thread ID
}

message TransactionEnd {
    int64 executeTime = 1;
    string transactionId = 2;
    repeated Pair props = 3;
}

message Pair {
    string key = 1;
    string value = 2;
}

enum EntryType {
    ENTRYTYPECOMPATIBLEPROTO2 = 0;
    TRANSACTIONBEGIN = 1;
    ROWDATA = 2;         // typical row changes
    TRANSACTIONEND = 3;
    HEARTBEAT = 4;       // internal heartbeat event
    GTIDLOG = 5;         // optional GTID log event
}

enum EventType {
    EVENTTYPECOMPATIBLEPROTO2 = 0;
    INSERT = 1;
    UPDATE = 2;
    DELETE = 3;
    CREATE = 4;
    ALTER = 5;
    ERASE = 6;
    QUERY = 7;
    TRUNCATE = 8;
    RENAME = 9;
    CINDEX = 10;    // CREATE INDEX
    DINDEX = 11;    // DROP INDEX
    GTID = 12;
    XACOMMIT = 13;
    XAROLLBACK = 14;
    MHEARTBEAT = 15; // Master heartbeat
}

enum Type {
    TYPECOMPATIBLEPROTO2 = 0;
    ORACLE = 1;
    MYSQL = 2;
    PGSQL = 3; // ...
}
```

### **Key Points about `CanalEntry.proto`**:

1. **`Entry`**: 
   - A single top-level binlog event in Canal’s format.  
   - The `EntryType` can indicate `TRANSACTIONBEGIN`, `ROWDATA`, `TRANSACTIONEND`, etc.  
   - The `storeValue` is usually another serialized message (e.g., `RowChange`) for row data changes, or `TransactionBegin`/`TransactionEnd` for transaction boundaries.

2. **`RowChange`**: 
   - Contains the **`eventType`** (INSERT, UPDATE, DELETE, etc.)  
   - A list of **`RowData`** items (each `RowData` has `beforeColumns` and `afterColumns`).  
   - If it’s a DDL statement, `isDdl = true` and the `sql` field contains the DDL command text.

3. **`Column`**: 
   - Describes a single column’s name, value, whether it changed (`updated`), whether it’s a primary key, etc.  

4. **`Header`**: 
   - Metadata about the binlog position (`logfileName`, `logfileOffset`), the table (`schemaName`, `tableName`), the server’s ID, etc.  

Thus, `CanalEntry.proto` is basically the **schema** for the **actual data** (the row changes and metadata) that Canal extracts from MySQL’s binlog.

---

# 3. `AdminPacket.proto`

This file is used for **administrative commands**—not the typical “data row” flow. For instance, you might control or inspect a running Canal instance: **start**, **stop**, **reload**, **list** instances, retrieve logs, etc.

```protobuf
syntax = "proto3";
package com.alibaba.otter.canal.protocol;

option java_package = "com.alibaba.otter.canal.protocol";
option java_outer_classname = "AdminPacket";
option optimize_for = SPEED;

enum PacketType {
    PACKAGETYPECOMPATIBLEPROTO2 = 0;
    HANDSHAKE = 1;
    CLIENTAUTHENTICATION = 2;
    ACK = 3;
    SERVER = 4;
    INSTANCE = 5;
    LOG = 6;
}

message Packet {
    oneof magic_number_present {
        int32 magic_number = 1;
    }
    oneof version_present {
        int32 version = 2;
    }
    PacketType type = 3;
    bytes body = 4;
}

message Ack {
    oneof error_code_present {
        int32 code = 1;
    }
    string message = 2;
}

message Handshake {
    oneof communication_encoding_present {
        string communication_encoding = 1;
    }
    bytes seeds = 2;
}

message ClientAuth {
    string username = 1;
    bytes password = 2;
    oneof net_read_timeout_present {
         int32 net_read_timeout = 3;
    }
    oneof net_write_timeout_present {
        int32 net_write_timeout = 4;
    }
}

/** 
 * Used for server-level commands: check, start, stop, restart, list.
 * The `action` field clarifies the admin command’s intention.
 */
message ServerAdmin {
    string action = 1; // e.g. "check", "start", "stop", "restart", "list"
}

/**
 * Used for instance-level actions on a particular 'destination':
 * The `action` could be "check", "start", "stop", "reload" etc.
 */
message InstanceAdmin {
    string destination = 1;
    string action = 2;
}

/**
 * Used for retrieving or tailing logs from Canal or from a particular instance.
 * - `type` could be "canal" or "instance"
 * - `action` might be something like "tail"
 * - Optionally specify a `destination`, `file`, and a `count` for the number of lines
 */
message LogAdmin {
    string type = 1;
    string action = 2;
    oneof destination_present {
        string destination = 3;
    }
    oneof file_present {
        string file = 4;
    }
    oneof count_present {
        int32 count = 5;
    }
}
```

### **Key Points about `AdminPacket.proto`**:
- Similar to `CanalPacket.proto`, it defines a `Packet` wrapper with an admin-specific `PacketType`.  
- Messages like `ServerAdmin`, `InstanceAdmin`, `LogAdmin` let you send commands to manage the Canal server or instances.

---

# 4. How It All Fits Together

1. **Connection & Authentication**  
   - The client connects to the Canal TCP port.  
   - The server sends `Packet(type=HANDSHAKE, body=Handshake)`.  
   - The client responds with `Packet(type=CLIENTAUTHENTICATION, body=ClientAuth)`.  
   - The server returns `Packet(type=ACK, body=Ack)` indicating success or failure.

2. **Subscription**  
   - The client sends a `Packet(type=SUBSCRIPTION, body=Sub)` telling the server which DB or tables to monitor (`filter`).  
   - The server typically replies with an `ACK` to confirm.

3. **Data Fetching**  
   - The client sends `Packet(type=GET, body=Get)` to request a batch of row changes.  
   - The server replies with `Packet(type=MESSAGES, body=Messages)` containing the binlog events.  
   - Each event in `Messages.messages[]` is typically a serialized `Entry` (from `CanalEntry.proto`).  
   - The client decodes each `Entry`, checks its `EntryType`, and if it’s `ROWDATA`, decodes the `storeValue` into a `RowChange`.

4. **Acknowledgment**  
   - After processing the batch, the client sends `Packet(type=CLIENTACK, body=ClientAck)` or `CLIENTROLLBACK` on failure.  
   - `batch_id` ensures both sides know which batch we’re referencing.

5. **Administration**  
   - Tools or admin clients can send `Packet(type=SERVER, body=ServerAdmin)` or `type=INSTANCE, body=InstanceAdmin)` to control Canal.  
   - The server responds with an `Ack` or other relevant message.

---

## In Summary

- **`CanalPacket.proto`** handles the **network-level messages** for standard CDC operations: handshake, subscription, pulling data, acknowledging batches, etc. 
- **`CanalEntry.proto`** defines the **structure of the binlog events** once parsed: the row changes, columns, transaction boundaries, and so on. This is the “payload” you actually process when you do CDC.  
- **`AdminPacket.proto`** is for **admin commands**, letting you start/stop or inspect the Canal server or an instance.  

**Conceptually**:
1. Canal wraps every request/response in a top-level `Packet`.  
2. The `PacketType` tells you how to parse the `body` (Handshake, ClientAuth, Messages, etc.).  
3. For binlog row data, you typically see `MESSAGES` packets, which contain one or more serialized `Entry` objects. Each `Entry` can hold row-change data (`RowChange`) or transaction markers, etc.  

This design allows Canal to separate **transport** (the top-level `Packet` with optional compression and a type) from the **domain model** (`RowChange`, `Column`, `TransactionBegin`, etc.). The result is a flexible and extensible protocol for streaming MySQL binlog changes—and optionally controlling the Canal server itself.