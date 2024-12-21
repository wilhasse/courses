## Overview of the `.proto` files

1. **`polarx.proto`**
   Defines the high-level message types (enums) for **client messages** and **server messages**, as well as some basic `Ok`/`Error` responses.
2. **`polarx_connection.proto`**
   Specifies messages related to capabilities negotiation and the basic open/close of connections.
3. **`polarx_datatypes.proto`**
   Provides the fundamental data type definitions (`Scalar`, `Object`, `Array`, `Any`) used throughout the protocol messages.
4. **`polarx_exec_plan.proto`**
   Defines messages that describe **execution plans** (e.g., table scans, key-based lookups, filter, aggregation, etc.), including single-table or multi-step plan operators.
5. **`polarx_expect.proto`**
   Defines an “expectation” mechanism to open/close certain states or conditions on the server side.
6. **`polarx_expr.proto`**
   Specifies how to represent expressions in PolarDB-X, including function calls, operators, placeholders, references, and more.
7. **`polarx_notice.proto`**
   Describes how the server sends **notices** or warnings back to the client, including session state changes.
8. **`polarx_physical_backfill.proto`**
   Contains messages used for *physical backfill*—managing files and data during DDL or other background tasks (file existence checks, copying data, disk info, etc.).
9. **`polarx_resultset.proto`**
   Defines how result sets and rows are returned from server to client, including chunked results, metadata columns, row formats, etc.
10. **`polarx_session.proto`**
    Defines session-oriented messages, such as authentication, session kill, session reset, etc.
11. **`polarx_sql.proto`**
    Provides messages related to executing SQL statements, preparing queries with parameters, returning success or error states, etc.

------

## 2. `polarx.proto` – Entry Point: Client/Server Messages

```
protobufCopiar códigomessage ClientMessages {
  enum Type {
    // Connection
    CON_CAPABILITIES_GET = 1;
    CON_CAPABILITIES_SET = 2;
    CON_CLOSE = 3;
    ...
    // Session
    SESS_AUTHENTICATE_START = 4;
    SESS_AUTHENTICATE_CONTINUE = 5;
    SESS_RESET = 6;
    SESS_CLOSE = 7;
    ...
    // SQL
    SQL_STMT_EXECUTE = 12;
    ...
    // CRUD
    CRUD_FIND = 17;
    CRUD_INSERT = 18;
    CRUD_UPDATE = 19;
    CRUD_DELETE = 20;
    ...
    EXEC_PLAN_READ = 100;
    EXEC_SQL = 101;
    ...
    SESS_NEW = 110;
    SESS_KILL = 111;
    ...
    GET_TSO = 113;
    AUTO_SP = 115;
    FILE_OPERATION_GET_FILE_INFO = 116;
    FILE_OPERATION_TRANSFER_FILE_DATA = 117;
    FILE_OPERATION_FILE_MANAGE = 118;
    ...
  }
}

message ServerMessages {
  enum Type {
    OK = 0;
    ERROR = 1;
    CONN_CAPABILITIES = 2;
    SESS_AUTHENTICATE_CONTINUE = 3;
    SESS_AUTHENTICATE_OK = 4;
    NOTICE = 11;
    ...
    RESULTSET_COLUMN_META_DATA = 12;
    RESULTSET_ROW = 13;
    RESULTSET_FETCH_DONE = 14;
    ...
    SQL_STMT_EXECUTE_OK = 17;
    ...
    RESULTSET_TOKEN_DONE = 19;
    RESULTSET_TSO = 20;
    ...
    RESULTSET_GET_FILE_INFO_OK = 22;
    RESULTSET_TRANSFER_FILE_DATA_OK = 23;
    RESULTSET_FILE_MANAGE_OK = 24;
  }
}

message Ok {
  optional string msg = 1;
}

message Error {
  optional Severity severity = 1 [default = ERROR];
  required uint32 code = 2;
  required string sql_state = 4;
  required string msg = 3;

  enum Severity {
    ERROR = 0;
    FATAL = 1;
  }
}
```

### Purpose

- **`ClientMessages.Type`** enumerates all possible message types the client can send to the server.
- **`ServerMessages.Type`** enumerates all possible message types the server can respond with.
- `Ok` and `Error` messages provide uniform success/error responses.

### Usage

When the client sends a message, it sets the message type (e.g., `SQL_STMT_EXECUTE`) and the corresponding payload (e.g., an instance of `StmtExecute` from `polarx_sql.proto`). The server responds with a type from `ServerMessages.Type` (e.g., `ERROR`, `OK`, or a `RESULTSET_ROW` stream).

------

## 3. `polarx_connection.proto` – Capabilities and Connection Management

```
protobufCopiar códigomessage Capability {
  required string name = 1;
  required PolarXRPC.Datatypes.Any value = 2;
}

message Capabilities {
  repeated Capability capabilities = 1;
}

message CapabilitiesGet {}

message CapabilitiesSet {
  required Capabilities capabilities = 1;
}

message Close {}
```

### Purpose

- **Capability** / **Capabilities**: The server and client negotiate capabilities (e.g., whether chunked result sets are supported, transaction features, etc.).
- **CapabilitiesGet/CapabilitiesSet**: The client asks the server for capabilities or sets them.
- **Close**: Terminates the connection (or the logical session on top of a connection).

### Usage

- On connecting, the client may send `CON_CAPABILITIES_GET` to see what the server supports, or `CON_CAPABILITIES_SET` to configure certain features (like enabling chunked responses).
- The server replies with `CONN_CAPABILITIES` (holding `Capabilities`).

------

## 4. `polarx_datatypes.proto` – Core Data Types

This file is crucial because many other messages reference these reusable data types:

```
protobufCopiar códigomessage Scalar {
  // Nested sub-messages for string & octets
  message String { ... }
  message Octets { ... }

  enum Type {
    V_SINT = 1;
    V_UINT = 2;
    V_NULL = 3;
    V_OCTETS = 4;
    V_DOUBLE = 5;
    V_FLOAT = 6;
    V_BOOL = 7;
    V_STRING = 8;
    V_PLACEHOLDER = 9;
    V_IDENTIFIER = 10;
    V_RAW_STRING = 11;
  }

  required Type type = 1;
  optional sint64 v_signed_int = 2;
  optional uint64 v_unsigned_int = 3;
  optional Octets v_octets = 5;
  optional double v_double = 6;
  optional float  v_float = 7;
  optional bool   v_bool = 8;
  optional String v_string = 9;
  optional uint32 v_position = 10;
  optional String v_identifier = 11;
}

message Object {
  message ObjectField {
    required string key = 1;
    required Any value = 2;
  }
  repeated ObjectField fld = 1;
}

message Array {
  repeated Any value = 1;
}

message Any {
  enum Type { SCALAR = 1; OBJECT = 2; ARRAY = 3; }
  required Type type = 1;

  optional Scalar scalar = 2;
  optional Object obj = 3;
  optional Array array = 4;
}

message SessionVariable {
  required string key = 1;
  required Scalar value = 2;
}
```

### Purpose

Defines a flexible, nested data structure to represent:

- Single values (integers, strings, booleans, placeholders, etc.).
- Composite values (objects, arrays).
- Session variables (key/value pairs).

### Usage

- Many messages need to pass parameters or expressions that can be typed at runtime, e.g., placeholders in a query or dynamic table names.
- `Scalar` is fundamental for representing single values (signed/unsigned int, double, string, etc.).
- `Any` can hold either `Scalar`, `Object`, or `Array`.
- `SessionVariable` is used to set or modify session-level parameters (like `autocommit=1`).

------

## 5. `polarx_exec_plan.proto` – Execution Plan Messages

PolarDB-X can send or receive an “exec plan” describing how to read from a table, apply filters, etc. It’s relevant in a distributed or multi-step scenario.

### Main Structures

1. **Plan Components**

   - `TableInfo`, `IndexInfo`, `Transaction`, `BloomFilter` (not shown in detail above).

   - Example:

     ```
     protobufCopiar códigomessage TableInfo {
       optional int64 version = 1;
       required PolarXRPC.Datatypes.Scalar name = 2;
       optional PolarXRPC.Datatypes.Scalar schema_name = 3;
     }
     ```

     This indicates which table (and optional schema) to operate on.

2. **Operations**

   - `GetPlan` and `RangeScan` define how to read rows by a key or a range.
   - `Filter`, `Project`, `TableProject`, `Aggr` define plan operators to refine or transform the data.

3. **`AnyPlan`**

   - Acts like a “union” type; it can hold one of several plan subtypes (GET, TABLE_SCAN, RANGE_SCAN, etc.).

4. **Top-Level `ExecPlan`**

   ```
   protobufCopiar códigomessage ExecPlan {
     optional Transaction transaction = 1;
     optional AnyPlan plan = 2;
     optional bytes plan_digest = 3;
     repeated PolarXRPC.Datatypes.Scalar parameters = 4;
     repeated PolarXRPC.Datatypes.SessionVariable session_variables = 5;
     optional int32 token = 6;
     optional bool reset_error = 7;
     ...
   }
   ```

   - Encapsulates the entire plan, plus any placeholders (`parameters`), session variables, or transaction context.

### Purpose

- Allows a client to describe how it wants to fetch or process data, possibly offloading the “plan” logic to the server engine.
- The server can interpret the plan, run sub-operations (like range scans, filters, or aggregates), and return results.

------

## 6. `polarx_expect.proto` – Expectation Handling

```
protobufCopiar códigomessage Open {
  message Condition {
    enum ConditionOperation {
      EXPECT_OP_SET = 0;
      EXPECT_OP_UNSET = 1;
    };
    required uint32 condition_key = 1;
    optional bytes condition_value = 2;
    optional ConditionOperation op = 3 [default = EXPECT_OP_SET];
  };

  enum CtxOperation {
    EXPECT_CTX_COPY_PREV = 0;
    EXPECT_CTX_EMPTY = 1;
  };
  optional CtxOperation op = 1 [default = EXPECT_CTX_COPY_PREV];
  repeated Condition cond = 2;
}

message Close {}
```

### Purpose

- Provides a way to set or unset “expectations” on the server side.
- “Conditions” might be used to check server state or configure an environment for the upcoming commands.

### Usage

- The client can “open” an expect block and set conditions (`cond`) that must be satisfied; if they aren’t, the server might fail or adjust behavior.
- “Close” ends that block of expectations.

------

## 7. `polarx_expr.proto` – Expression Trees

```
protobufCopiar códigomessage Expr {
  enum Type {
    IDENT = 1;
    LITERAL = 2;
    VARIABLE = 3;
    FUNC_CALL = 4;
    OPERATOR = 5;
    PLACEHOLDER = 6;
    OBJECT = 7;
    ARRAY = 8;
    REF = 9;
  };
  required Type type = 1;

  // Depending on the Type, one of these is used:
  optional ColumnIdentifier identifier = 2;
  optional string       variable = 3;
  optional PolarXRPC.Datatypes.Scalar literal = 4;
  optional FunctionCall function_call = 5;
  optional Operator     operator = 6;
  optional uint32       position = 7;
  optional Object       object = 8;
  optional Array        array = 9;
  optional uint32       ref_id = 10;
}

// ... plus ColumnIdentifier, FunctionCall, Operator, ...
```

### Purpose

- An `Expr` node can be an identifier (column name), a literal, a function call, or an operator invocation.
- Useful for advanced filtering, computed columns, or extended functionalities.

### Usage

- In `polarx_exec_plan.proto`, operators like `Filter` or `Aggr` reference these expressions.
- For example, an `Aggr` might have `expr = SUM(column_name)`. Internally, that’s represented here.

------

## 8. `polarx_notice.proto` – Notices / Warnings

```
protobufCopiar códigomessage Frame {
  enum Scope {
    GLOBAL = 1;
    LOCAL = 2;
  };
  enum Type {
    WARNING = 1;
    SESSION_VARIABLE_CHANGED = 2;
    SESSION_STATE_CHANGED = 3;
    GROUP_REPLICATION_STATE_CHANGED = 4;
    SERVER_HELLO = 5;
  };
  required uint32 type = 1;
  optional Scope  scope = 2 [default = GLOBAL];
  optional bytes payload = 3;
}

message Warning {
  enum Level {
    NOTE = 1;
    WARNING = 2;
    ERROR = 3;
  };
  optional Level  level = 1 [default = WARNING];
  required uint32 code = 2;
  required string msg = 3;
}

// ...
```

### Purpose

- A “notice” is a non-critical message or warning from the server.
- `SessionStateChanged` or `SessionVariableChanged` might notify clients that something changed in the environment (e.g., `CURRENT_SCHEMA` changed).

### Usage

- Typically piggybacked along with result sets or in between messages to let the client know about server-side events.
- The server frames these notices, and the client can display or act on them.

------

## 9. `polarx_physical_backfill.proto` – Physical File Management

```
protobufCopiar códigomessage TableInfo {
  required string table_schema = 1;
  required string table_name = 2;
  required bool partitioned = 3;
  repeated string physical_partition_name = 4;
  repeated FileInfo file_info = 5;
}

message FileInfo {
  required string directory = 1;
  required string file_name = 2;
  required string partition_name = 3 [default = ""];
  optional uint64 data_size = 4;
  optional bool existence = 5;
  optional bool temp_file = 6;
}

message GetFileInfoOperator {
  enum Type {
    CHECK_SRC_FILE_EXISTENCE = 1;
    CHECK_TAR_FILE_EXISTENCE = 2;
    ...
    GET_SRC_DIR_FREE_SPACE = 5;
    ...
  };
  required Type operator_type = 1;
  optional TableInfo table_info = 2;
  optional DiskInfo disk_info = 3;
}
```

### Purpose

- Manages data/backfill tasks such as copying/validating physical files, checking disk info, or ensuring that certain table partitions exist.
- Typical use case: **online DDL** or distributed partition management where raw data files (like `.ibd` for InnoDB) need to be manipulated.

### Usage

- The client or internal engine sends a `FILE_OPERATION_*` request with the appropriate operator to check file existence, get free space, copy files, etc.
- The server responds with success/failure messages (like `RESULTSET_GET_FILE_INFO_OK`).

------

## 10. `polarx_resultset.proto` – Returning Query Results

```
protobufCopiar códigomessage ColumnMetaData {
  enum FieldType { SINT = 1; UINT = 2; DOUBLE = 5; ... }
  enum OriginalType { MYSQL_TYPE_DECIMAL = 0; MYSQL_TYPE_TINY = 1; ... }

  required FieldType type = 1;
  required OriginalType original_type = 2;
  optional bytes name = 3;
  ...
}

message Row {
  repeated bytes field = 1;
}

message Chunk {
  required uint32 row_count = 1;
  repeated Column columns = 2;
}

message Column {
  optional bytes null_bitmap = 1;
  optional FixedSizeColumn fixed_size_column = 2;
  optional VariableSizeColumn variable_size_column = 3;
}
```

### Purpose

- Defines how the server streams back rows.
- **ColumnMetaData** describes each column’s type, name, etc.
- **Row** uses a repeated `bytes field` for each column.
- **Chunk** is an optional chunked streaming format for large result sets. Instead of sending row-by-row, the server can batch multiple columns.

### Usage

- After a 

  ```
  SQL_STMT_EXECUTE
  ```

   or 

  ```
  EXEC_PLAN_READ
  ```

  , the server returns one or more:

  - `RESULTSET_COLUMN_META_DATA` messages to define column schemas.
  - `RESULTSET_ROW` messages or `Chunk` messages to hold data.
  - `RESULTSET_FETCH_DONE` (end of data).

------

## 11. `polarx_session.proto` – Session Commands

```
protobufCopiar códigomessage AuthenticateStart {
  required string mech_name = 1;
  optional bytes auth_data = 2;
  optional bytes initial_response = 3;
}

message AuthenticateContinue {
  required bytes auth_data = 1;
}

message AuthenticateOk {
  optional bytes auth_data = 1;
}

message KillSession {
  enum KillType {
    QUERY = 1;
    CONNECTION = 2;
  }
  required KillType type = 1;
  required uint64 x_session_id = 2;
}

message Reset {}
message Close {}
```

### Purpose

- These messages handle session lifecycle:
  - `AuthenticateStart/Continue/Ok` for the challenge-response authentication flow.
  - `KillSession` to kill a query or a connection.
  - `Reset` or `Close` to reset or close the session context.

### Usage

- Tied to `SESS_AUTHENTICATE_START`, `SESS_AUTHENTICATE_CONTINUE`, `SESS_AUTHENTICATE_OK`, etc. in `ClientMessages.Type`.
- The server replies with `SESS_AUTHENTICATE_CONTINUE`, `SESS_AUTHENTICATE_OK`, or an error.

------

## 12. `polarx_sql.proto` – Executing SQL Statements

```
protobufCopiar códigomessage StmtExecute {
  optional string namespace = 3 [default = "sql"];
  optional bytes stmt = 1;
  optional bytes stmt_digest = 12;
  optional bytes hint = 13;
  optional bool chunk_result = 14 [default = false];
  ...
  repeated PolarXRPC.Datatypes.Any args = 2; 
  ...
  optional string schema_name = 5;
  repeated PolarXRPC.Datatypes.SessionVariable session_variables = 6;
  ...
}

message StmtExecuteOk {}

message TokenOffer {
  optional int32 token = 1 [default = -1];
}
```

### Purpose

- **`StmtExecute`** is the main message for issuing a SQL query. It can include:
  - The raw SQL text (`stmt`)
  - A statement digest for caching or monitoring
  - Execution “hint” metadata
  - Optional chunked result enabling
  - Bound parameters (`args`)
  - Session overrides (schema, variables)
  - Transaction control fields (snapshot_seq, commit_seq, etc.)
- **`StmtExecuteOk`** indicates a successful statement execution (like an `OK` packet in classic MySQL protocol).

### Usage

- The client sends `SQL_STMT_EXECUTE` as the message type, with payload `StmtExecute`.
- The server returns `SQL_STMT_EXECUTE_OK` or an error, plus result set messages if it’s a `SELECT`.

------

# Putting It All Together

1. **Connection Initialization**
   - The client may start by sending `CON_CAPABILITIES_GET` or `SESS_AUTHENTICATE_START`.
   - Server responds with `CONN_CAPABILITIES`, `SESS_AUTHENTICATE_CONTINUE`, or `SESS_AUTHENTICATE_OK` as needed.
2. **Executing Queries**
   - The client sends `SQL_STMT_EXECUTE` with a `StmtExecute` message.
   - The server streams back `RESULTSET_COLUMN_META_DATA` and multiple `RESULTSET_ROW` (or `Chunk`) messages until `RESULTSET_FETCH_DONE` or `SQL_STMT_EXECUTE_OK`.
3. **Distributed or Advanced Plans**
   - The client may send `EXEC_PLAN_READ` or `EXEC_SQL` with an `ExecPlan` message.
   - The engine interprets the plan (`AnyPlan`) and returns data similarly via result set messages.
4. **File Operations or Physical Backfill**
   - The client (or a management node) can send `FILE_OPERATION_GET_FILE_INFO` or `FILE_OPERATION_TRANSFER_FILE_DATA` messages with the corresponding operators.
   - The server returns success/failure messages (e.g., `RESULTSET_GET_FILE_INFO_OK`).
5. **Session Maintenance**
   - The client can `SESS_RESET`, `SESS_CLOSE`, or `SESS_KILL`.
   - The server acknowledges with `Ok` or `Error`.
6. **Notices and Warnings**
   - The server can asynchronously send `NOTICE` frames about warnings, session changes, etc.

------

## Conclusion

The PolarDB-X protobuf definitions provide a **full-featured, structured protocol** for:

- Negotiating capabilities
- Managing sessions and authentication
- Sending and receiving queries/plans/CRUD
- Returning row-based or chunk-based results
- Managing distributed or advanced storage-level operations
- Handling notices, warnings, or backfill tasks

Each `.proto` file focuses on a different aspect of the communication. They interconnect primarily via shared data types defined in `polarx_datatypes.proto` and the `ClientMessages`/`ServerMessages` enums in `polarx.proto`.

Overall, these definitions allow PolarDB-X clients (or internal components) to talk to PolarDB-X servers or nodes in a well-defined, binary-encoded, forward/backward compatible manner.

# Compare to Mysql X Plugin

The PolarDB-X protocol and the MySQL X Plugin protocol are both defined using Protocol Buffers (`proto2`) and are conceptually similar in that they facilitate client-server communication for database operations, including capabilities negotiation, session control, CRUD operations, and execution of SQL statements. Both are designed for a modern, asynchronous and extensible communication layer, going beyond the older text-based protocols. However, there are notable differences stemming from the different objectives, architectural constraints, and feature sets of PolarDB-X and MySQL X.

Below is a detailed comparison:

------

## 1. Shared Concepts

Both protocols share a foundational approach:

1. **Protobuf-based**:
   Both use `syntax = "proto2"` and define a variety of message types (e.g. `ClientMessages`, `ServerMessages`) and data structures (`Scalar`, `Any`, `Expr`), providing a binary, schema-driven protocol that can evolve while preserving backward compatibility.
2. **Message Categories**:
   - **Connection and Capabilities**: Both define messages for `CON_CAPABILITIES_GET`, `CON_CAPABILITIES_SET`, and returning a `CONN_CAPABILITIES` message.
   - **Session Management**: Both handle sessions, including authentication (`SESS_AUTHENTICATE_START`, `SESS_AUTHENTICATE_CONTINUE`, `SESS_AUTHENTICATE_OK`) and session-level commands (`SESS_RESET`, `SESS_CLOSE`).
   - **Error/OK Handling**: Both define `Ok` and `Error` messages with a similar structure (an error code, SQL state, message, severity).
3. **CRUD and SQL Execution**:
   Both define messages for executing SQL statements (`SQL_STMT_EXECUTE`) and returning results (`RESULTSET_COLUMN_META_DATA`, `RESULTSET_ROW`, `RESULTSET_FETCH_DONE`, etc.). They also define CRUD operations (`FIND`, `INSERT`, `UPDATE`, `DELETE`) for document-oriented or table-oriented data models.
4. **Notices/Warnings and State Changes**:
   Both protocols have a `NOTICE` message framework to inform clients of warnings, session variable changes, and other non-critical signals from the server.
5. **Data Types**:
   They share very similar definitions of `Scalar`, `Object`, `Array`, `Any`, and `Expr`, including the idea of `DocumentPathItem`, `Identifier`, `FunctionCall`, and `Operator`. The type systems are nearly identical, with `Scalar` supporting integers, floats, doubles, booleans, strings, octets, and null.

------

## 2. Differences in Scope and Feature Set

While MySQL X Plugin is designed as a general-purpose extension to MySQL’s capabilities—especially focusing on document store functionality and a unified protocol for both SQL and CRUD—PolarDB-X’s protocol includes features and message types tailored to the distributed, sharded environment and advanced transactional scenarios.

**a. Execution Plan vs. Simple CRUD/SQL**

- **PolarDB-X**:
  The PolarDB-X protocol includes a `polarx_exec_plan.proto`, which describes distributed execution plans (`ExecPlan` and `AnyPlan` with subplan types like `GetPlan`, `RangeScan`, `Filter`, `Aggr`, etc.). This reflects PolarDB-X’s need to push down complex execution logic and multi-step plans to compute nodes.
- **MySQL X Plugin**:
  The MySQL X protocol focuses more on direct operations: SQL queries, CRUD commands (document store operations), prepared statements, and cursors. It doesn't define an execution plan language for pushing complex distributed execution instructions. Instead, it handles more conventional queries, views, and basic CRUD. Complex execution strategy typically remains internal to MySQL and is not exposed as a first-class protocol concept.

**b. Distributed Transactional Features**

- **PolarDB-X**:
  PolarDB-X includes specialized fields for distributed transaction management, such as `snapshot_seq`, `commit_seq`, `use_cts_transaction`, `token`, and `mark_distributed`. There are also `GetTSO` and `ResultTSO` messages for TSO (timestamp oracle) operations, as well as `AUTO_SP` (auto savepoint) functionalities. These messages support its distributed transaction model.
- **MySQL X Plugin**:
  MySQL X generally assumes a single-server transactional model. While MySQL Group Replication and other distributed features exist, the X protocol doesn’t expose explicit distributed transaction or global TSO fields at this level. Transactional commands are implicit (like `BEGIN`, `COMMIT`, `ROLLBACK`) rather than pushing advanced distributed transaction parameters via the protocol.

**c. File and Backfill Operations**

- **PolarDB-X**:
  Provides `FILE_OPERATION_GET_FILE_INFO`, `FILE_OPERATION_TRANSFER_FILE_DATA`, and `FILE_OPERATION_FILE_MANAGE` message types and related structures (`polarx_physical_backfill.proto`). These are used for managing physical tablespace data, IBD files, and disk info in a distributed environment. This is a low-level feature enabling DDL migrations, physical backfill of data, and cross-node file operations.
- **MySQL X Plugin**:
  MySQL X does not define operations for direct file handling in its protocol. Physical file operations are not part of its scope. It focuses on logical CRUD and SQL operations.

**d. Extra Operations in MySQL X**

- MySQL X Plugin

  :

  Includes some features not explicitly shown in PolarDB-X’s proto definitions, such as:

  - `CRUD_CREATE_VIEW`, `CRUD_MODIFY_VIEW`, `CRUD_DROP_VIEW`: direct protocol-level instructions to manage views.
  - `PREPARE_PREPARE`, `PREPARE_EXECUTE`, `PREPARE_DEALLOCATE`: built-in server-side prepare/execute operations, more aligned with client-side prepared statements.
  - `CURSOR_OPEN`, `CURSOR_FETCH`, `CURSOR_CLOSE`: explicit cursor operations over prepared statements.

  PolarDB-X does not expose the same level of built-in cursor or view management in its protocol. Instead, these are presumably handled via SQL or other integrated mechanisms.

**e. Expect Blocks and Conditions**

- Both protocols define `Expect` operations. However, MySQL X’s `Expect` conditions (`EXPECT_NO_ERROR`, `EXPECT_FIELD_EXIST`, `EXPECT_DOCID_GENERATED`) are tailored towards ensuring certain conditions hold for subsequent operations, largely in the context of the document store. PolarDB-X’s `expect.proto` is conceptually similar but may integrate differently with distributed execution environments or other specialized conditions.

**f. Metadata Differences**

- Both define similar `ColumnMetaData`, but PolarDB-X includes `original_flags` and other fields specific to its engine. MySQL X metadata is closely modeled on MySQL’s internal definitions. PolarDB-X also supports some additional or different fields due to handling a distributed system.

**g. Additional Content Types**

- MySQL X’s `ContentType_BYTES` and `ContentType_DATETIME` enumerations are more closely aligned to MySQL’s data formats (like JSON, GEOMETRY). PolarDB-X focuses more on the distributed execution plan aspects and specialized placeholders for parameters and identifiers.

------

## 3. Architectural Intent

- **PolarDB-X Protocol**:
  Designed specifically for PolarDB-X, a distributed relational database system, it integrates deeply with multi-node execution plans, snapshot reads, global transaction sequencing, and physical file operations. The protocol aims to handle the complexity of distributing and orchestrating queries, scans, and data manipulation across multiple shards/nodes.
- **MySQL X Plugin Protocol**:
  Created as an extensible, next-generation protocol for MySQL, emphasizing a unified data model (documents and relational tables), CRUD operations, asynchronous communications, and enhanced client-server interactions without necessarily exposing internal planning logic or distributed transaction details. It remains more “logical” and high-level, leaving details like distributed execution strategies inside the server.

------

## 4. Evolution and Extensibility

Both protocols can be extended. However:

- **PolarDB-X**:
  Has evolved to include fields and operations critical to its distributed nature—like `ExecPlan`, `GetTSO`, `AutoSp`—not found in MySQL X. This shows PolarDB-X’s specialization and possibly less concern for generic backward compatibility with “standard” MySQL X, since it’s tailored to a specific ecosystem.
- **MySQL X Plugin**:
  Is intended as a broad-facing client protocol for a large user community, with stable versioning and careful evolution to maintain compatibility and cater to both relational and document-store use cases in MySQL.

------

## Summary of Key Differences

- **Distributed Execution Plans**: PolarDB-X explicitly models execution plans in its protocol, MySQL X does not.
- **Distributed Transactions and TSO**: PolarDB-X includes fields and messages to handle distributed transactions and TSO. MySQL X does not.
- **Physical File Operations**: PolarDB-X supports physical backfill and file management. MySQL X does not.
- **Prepared Statements and Cursors**: MySQL X has built-in message types for prepared statements and cursors. PolarDB-X’s posted definitions do not.
- **View Management**: MySQL X supports direct CRUD-like messages for views; PolarDB-X does not.
- **Focus**: PolarDB-X’s protocol is more specialized to distributed database operations, while MySQL X is a general-purpose extension designed to unify SQL and NoSQL/document operations under the MySQL umbrella.

------

**In essence**: While PolarDB-X and MySQL X share a similar fundamental design approach and some message types, PolarDB-X extends the protocol to handle the complexities of a distributed SQL engine with execution plans and transaction coordination, whereas MySQL X focuses on providing a unified, flexible, document/relational protocol for a standalone or replicated MySQL instance.