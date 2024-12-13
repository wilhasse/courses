# Main

Below is an in-depth comparison of the two provided protobuf protocols: **MySQL X Protocol (mysqlx.proto)** and the **PolarDB-X protocol (polarx.proto)**. While both are conceptually similar—each defines a set of message types and structures used for communication between client and server—there are notable differences in their namespace, message sets, extended features, and intended use cases.

## 1. Namespace and Package Names

- MySQL X Protocol (mysqlx.proto)

  - Uses the package: `package Mysqlx;`
  - Java package: `option java_package = "com.mysql.cj.x.protobuf";`

- PolarDB-X Protocol (polarx.proto)

  - Uses the package: `package PolarXRPC;`
  - Java package: `option java_package = "com.mysql.cj.x.protobuf";`

Both protocols share the same Java package for compiled protobuf classes, but the PolarDB-X proto is under `PolarXRPC` rather than `Mysqlx`. This reflects a fork or extension of the MySQL X protocol intended for PolarDB-X’s environment.

## 2. Licensing and Origins

- **MySQL X Protocol**:
  - Developed by Oracle/MySQL.
  - Licensed under the GNU General Public License v2 with specific additional permissions.
  - Mentions MySQL server components, references official MySQL plugin and server structure.
- **PolarDB-X Protocol**:
  - Developed or customized by Alibaba for PolarDB-X.
  - Also uses the GPL v2 license with similar conditions and references to OpenSSL and other components.
  - Incorporates large portions of Oracle’s original protocol but clearly extends it with custom message types and functionality.

## 3. Syntax and Format

Both files use:

- `syntax = "proto2";`
- Similar style and coding conventions as per Google’s protobuf style guidelines.

They also follow the MySQL X Protocol pattern: a leading length, message type, and protobuf-encoded payload. Structurally, both define `ClientMessages` and `ServerMessages` enums for message type constants and `Ok` and `Error` messages to signal results and error conditions.

## 4. Core Message Types and Structure

### Shared Concepts

**Both protocols define:**

- A `ClientMessages` enumeration describing which messages can be sent from client to server.

- A `ServerMessages` enumeration describing which messages can be sent from server to client.

- ```
  Ok and Error
  ```
   messages that have similar semantics:

  - `Ok` is a generic acknowledgment message.
  - `Error` includes severity (ERROR or FATAL), a code, SQL state, and a human-readable message. When severity is FATAL, the client should close the connection.

The messages like `CON_CAPABILITIES_GET`, `CON_CAPABILITIES_SET`, `CON_CLOSE`, `SESS_AUTHENTICATE_START`, `SESS_AUTHENTICATE_CONTINUE`, `SESS_RESET`, `SESS_CLOSE`, `SQL_STMT_EXECUTE`, and CRUD operations (`CRUD_FIND`, `CRUD_INSERT`, `CRUD_UPDATE`, `CRUD_DELETE`) are present in both protocols. These form the common baseline of the MySQL X Protocol for session and statement management.

### Differences in the Client Message Set

**MySQL X Protocol (mysqlx.proto) ClientMessages:**

```
protobufCopiar códigoenum Type {
  CON_CAPABILITIES_GET = 1;
  CON_CAPABILITIES_SET = 2;
  CON_CLOSE = 3;
  SESS_AUTHENTICATE_START = 4;
  SESS_AUTHENTICATE_CONTINUE = 5;
  SESS_RESET = 6;
  SESS_CLOSE = 7;
  SQL_STMT_EXECUTE = 12;
  CRUD_FIND = 17;
  CRUD_INSERT = 18;
  CRUD_UPDATE = 19;
  CRUD_DELETE = 20;
  EXPECT_OPEN = 24;
  EXPECT_CLOSE = 25;
  CRUD_CREATE_VIEW = 30;
  CRUD_MODIFY_VIEW = 31;
  CRUD_DROP_VIEW = 32;
  PREPARE_PREPARE = 40;
  PREPARE_EXECUTE = 41;
  PREPARE_DEALLOCATE = 42;
  CURSOR_OPEN = 43;
  CURSOR_CLOSE = 44;
  CURSOR_FETCH = 45;
  COMPRESSION = 46;
}
```

**PolarDB-X Protocol (polarx.proto) ClientMessages:**

```
protobufCopiar códigoenum Type {
  CON_CAPABILITIES_GET = 1;
  CON_CAPABILITIES_SET = 2;
  CON_CLOSE = 3;
  SESS_AUTHENTICATE_START = 4;
  SESS_AUTHENTICATE_CONTINUE = 5;
  SESS_RESET = 6;
  SESS_CLOSE = 7;
  SQL_STMT_EXECUTE = 12;
  CRUD_FIND = 17;
  CRUD_INSERT = 18;
  CRUD_UPDATE = 19;
  CRUD_DELETE = 20;
  EXPECT_OPEN = 24;
  EXPECT_CLOSE = 25;

  // Additional messages unique to PolarDB-X:
  EXEC_PLAN_READ = 100;
  EXEC_SQL = 101;
  SESS_NEW = 110;
  SESS_KILL = 111;
  TOKEN_OFFER = 112;
  GET_TSO = 113;
  AUTO_SP = 115;
  FILE_OPERATION_GET_FILE_INFO = 116;
  FILE_OPERATION_TRANSFER_FILE_DATA = 117;
  FILE_OPERATION_FILE_MANAGE = 118;

  MSG_MAX = 127;
}
```

**Key Observations:**

- PolarDB-X extends the protocol with numerous additional client message types:
  - **EXEC_PLAN_READ (100)** and **EXEC_SQL (101)**: Suggests advanced capabilities for plan-based execution and raw SQL execution beyond the standard `SQL_STMT_EXECUTE`.
  - **SESS_NEW (110)** and **SESS_KILL (111)**: Introduce session creation and killing semantics that are not present in the vanilla MySQL X Protocol.
  - **TOKEN_OFFER (112)**: Possibly used for authentication or security token negotiation.
  - **GET_TSO (113)**: Likely related to fetching a Timestamp Oracle (TSO), relevant in distributed systems and multi-region transactions.
  - **AUTO_SP (115)**: Could be related to auto-stored procedures or some specialized logic.
  - **FILE_OPERATION_* (116, 117, 118)**: Indicates the ability to manage files (get info, transfer data, manage) via the protocol, which is not a standard part of MySQL X Protocol.

These additions reflect PolarDB-X’s distributed transaction needs, cluster management, and possibly some operational maintenance features like file operations.

### Differences in the Server Message Set

**MySQL X Protocol (mysqlx.proto) ServerMessages:**

```
protobufCopiar códigoenum Type {
  OK = 0;
  ERROR = 1;
  CONN_CAPABILITIES = 2;
  SESS_AUTHENTICATE_CONTINUE = 3;
  SESS_AUTHENTICATE_OK = 4;
  NOTICE = 11;
  RESULTSET_COLUMN_META_DATA = 12;
  RESULTSET_ROW = 13;
  RESULTSET_FETCH_DONE = 14;
  RESULTSET_FETCH_SUSPENDED = 15;
  RESULTSET_FETCH_DONE_MORE_RESULTSETS = 16;
  SQL_STMT_EXECUTE_OK = 17;
  RESULTSET_FETCH_DONE_MORE_OUT_PARAMS = 18;
  COMPRESSION = 19;
}
```

**PolarDB-X Protocol (polarx.proto) ServerMessages:**

```
protobufCopiar códigoenum Type {
  OK = 0;
  ERROR = 1;
  CONN_CAPABILITIES = 2;
  SESS_AUTHENTICATE_CONTINUE = 3;
  SESS_AUTHENTICATE_OK = 4;
  NOTICE = 11;
  RESULTSET_COLUMN_META_DATA = 12;
  RESULTSET_ROW = 13;
  RESULTSET_FETCH_DONE = 14;
  RESULTSET_FETCH_SUSPENDED = 15;
  RESULTSET_FETCH_DONE_MORE_RESULTSETS = 16;
  SQL_STMT_EXECUTE_OK = 17;
  RESULTSET_FETCH_DONE_MORE_OUT_PARAMS = 18;

  // Additional server-side result messages for PolarDB-X:
  RESULTSET_TOKEN_DONE = 19;
  RESULTSET_TSO = 20;
  RESULTSET_CHUNK = 21;
  RESULTSET_GET_FILE_INFO_OK = 22;
  RESULTSET_TRANSFER_FILE_DATA_OK = 23;
  RESULTSET_FILE_MANAGE_OK = 24;
}
```

**Key Observations:**

- PolarDB-X introduces:
  - **RESULTSET_TOKEN_DONE (19)**, possibly for signaling the completion of token-based operations.
  - **RESULTSET_TSO (20)**, likely returning timestamp or global ordering information for distributed transactions.
  - **RESULTSET_CHUNK (21)**, which might support large data sets or partial data transfer (streaming chunks).
  - **RESULTSET_GET_FILE_INFO_OK (22)**, **RESULTSET_TRANSFER_FILE_DATA_OK (23)**, **RESULTSET_FILE_MANAGE_OK (24)**, all aligned with the additional file operations defined on the client side.

These new server messages correlate directly to the extended client message capabilities, facilitating advanced features like file handling and distributed transaction support.

## 5. Message Metadata, Options, and Extensions

- MySQL’s original proto uses extensions to associate message types with `client_message_id` or `server_message_id` in combination with `google/protobuf/descriptor.proto`.

- PolarDB-X’s provided snippet does not show these protobuf descriptor extensions. It may or may not use them separately. The sample does not show 

  ```
  extend google.protobuf.MessageOptions
  ```

  used in polarx.proto, while mysqlx.proto does:

  ```
  protobufCopiar código// ifndef PROTOBUF_LITE
  extend google.protobuf.MessageOptions {
    optional ClientMessages.Type client_message_id = 100001;
    optional ServerMessages.Type server_message_id = 100002;
  }
  // endif
  ```

This difference suggests that MySQL X Protocol’s reference implementation might rely on these metadata extensions for message mapping, whereas PolarDB-X may rely on internal logic or a different approach.

## 6. High-Level Conceptual Differences

**MySQL X Protocol** is a general-purpose protocol for communicating with MySQL’s X Plugin, focusing on:

- CRUD operations
- SQL execution
- Session management
- Prepared statements
- Notifications and notices

**PolarDB-X Protocol** appears to build upon the MySQL X Protocol but is tailored towards a more complex, possibly distributed environment. It adds:

- Session creation and termination beyond the standard `SESS_CLOSE`.
- Token-based authentication or security tokens (`TOKEN_OFFER` and `RESULTSET_TOKEN_DONE`).
- Distributed transaction support through TSO (Timestamp Oracle) retrieval (`GET_TSO` and `RESULTSET_TSO`).
- Execution plan reads (`EXEC_PLAN_READ`), suggesting advanced internal performance or optimization features.
- Additional file operations to manage files remotely (`FILE_OPERATION_*`).

In essence, PolarDB-X extends the original protocol to support features crucial in distributed and cloud-native scenarios that go beyond what MySQL’s stock X Protocol provides.

## 7. Error and Ok Messages

Both protocols define `Ok` and `Error` messages similarly, with nearly identical fields and semantics. The `Error` message includes:

- `severity` (ERROR, FATAL)
- `code`
- `sql_state`
- `msg`

PolarDB-X retains this format and semantics, ensuring compatibility with the core handling of errors defined by MySQL X Protocol.

------

**In summary**, the primary difference is that `mysqlx.proto` represents the standard MySQL X Protocol focused on standard database operations and sessions, while `polarx.proto` is an extended variant (for PolarDB-X) that adds new message types and functionalities to accommodate advanced features like distributed transaction management, token operations, specialized SQL execution flows, and file management capabilities. The PolarDB-X proto builds on the original design but customizes and extends it to meet the requirements of a more complex, distributed database environment.

# Resultset

Below is a detailed comparative analysis of the MySQL X Protocol’s `mysqlx_resultset.proto` and the PolarDB-X `polarx_resultset.proto` definitions. While both describe the structure of resultsets returned from executing statements, PolarDB-X extends and modifies the protocol to support more advanced or custom features, especially around how data is transferred and represented.

## 1. Overall Structure and Commonalities

**Shared Concepts:**

- Both protocols define messages that represent a sequence of resultsets returned by the server after executing queries.
- They share similar concepts of:
  - **ColumnMetaData**: Describing the metadata of columns (type, length, flags, charset, etc.).
  - **Row**: Representing a row of data as a list of encoded field values.
  - **FetchDone**, **FetchDoneMoreResultsets**, and **FetchDoneMoreOutParams**: Signaling the end of a resultset or indicating that more resultsets or OUT parameter sets follow.

Thus, both follow the general MySQL X Protocol pattern, but PolarDB-X adds enhancements and additional structures for more complex data handling.

## 2. Differences in Message Definitions

### Additional Messages in PolarDB-X

**PolarXRPC.Resultset.Chunk and Column Messages**
PolarDB-X introduces a concept of **Chunk** and **Column** messages, which are not present in the MySQL X Protocol version shown:

- **Chunk**: Represents one chunk of a result set, containing a `row_count` and repeated `Column` messages. This indicates a possibly different or more efficient way to transmit large result sets. Instead of sending rows as separate messages (as MySQL X typically does with `Row` messages), PolarDB-X can send "chunks" that contain multiple columns of multiple rows in a potentially more compact or column-oriented format.
- **Column** (within a `Chunk`):
  Each `Column` in a chunk has:
  - A `null_bitmap` to indicate which rows have NULL values for that particular column.
  - Either a `FixedSizeColumn` or `VariableSizeColumn` value encoding, which allows more efficient representation based on data type and size.

This chunked, columnar approach is a significant departure from the traditional row-by-row `Row` message structure used in MySQL X Protocol. It suggests that PolarDB-X may support more optimal data transfer methods, batch encoding, or different retrieval patterns that are better suited for distributed or large-scale systems.

**TokenDone Message**
PolarDB-X defines a `TokenDone` message that includes a `token_left` field. This message type is not present in MySQL X’s `Resultset` definitions. It likely relates to the token-based flow control or session/transactional tokens defined in the PolarDB-X extensions.

### Modified and Extended Messages

**FetchDone Message**

- **MySQL X Protocol**: `FetchDone` is an empty message signaling that all resultsets are finished.

- PolarDB-X

  ```
  FetchDone
  ```

 includes additional optional fields:

  - `examined_row_count`: Potentially provides diagnostic or performance information about how many rows the server examined to produce the result.
  - `chosen_index`: Possibly indicating which index or execution plan variant was chosen.

These extra fields suggest PolarDB-X returns more query execution metadata to the client, which can be useful for optimization or client-side decision-making.

### ColumnMetaData Enhancements

**In MySQL X Protocol (mysqlx_resultset.proto):**

- `ColumnMetaData` defines basic fields: type, name, original_name, table, original_table, schema, catalog, collation, fractional_digits, length, flags, content_type.
- `FieldType` is defined with a subset of known MySQL data types, mapped to generic `SINT`, `UINT`, `DOUBLE`, `FLOAT`, `BYTES`, `TIME`, `DATETIME`, `SET`, `ENUM`, `BIT`, `DECIMAL`.

**In PolarDB-X (polarx_resultset.proto):**

- `ColumnMetaData` also includes `original_type` (of enum `OriginalType`) and `original_flags`. This provides a direct link back to MySQL’s internal type system, including the MySQL original type codes like `MYSQL_TYPE_VAR_STRING`, `MYSQL_TYPE_JSON`, etc.
- The inclusion of `original_type` and `original_flags` gives the client more detailed metadata, potentially useful for compatibility, debugging, or reconstructing the exact original schema semantics.
- Similar to MySQL X, it supports the same generic `FieldType` values, but with the added richness of original MySQL type info.

## 3. Data Encoding Differences

**MySQL X Protocol:**

- Uses `Row` messages containing repeated `bytes field` values for each column. Each `Row` maps directly to one logical row of data.
- Does not define a columnar encoding or a chunk-based approach. Rows arrive sequentially, each row a separate message.

**PolarDB-X:**

- Retains the 

  ```
  Row
  ```

   message (for compatibility or certain use cases) but introduces a new, more complex encoding approach:

  - Chunks and Columns

    : This suggests a column-oriented streaming of data. Each Column

    includes:

    - `null_bitmap` to indicate null values for multiple rows in a compressed manner.
    - `fixed_size_column` or `variable_size_column` for the actual column values. Fixed-size columns are optimal for numeric and fixed-length data types, while variable-size columns handle strings, BLOBs, etc.

This design allows PolarDB-X to send large volumes of data more efficiently, possibly reducing overhead in large result sets. It also may allow for partial reading of columns or more efficient distribution in a clustered environment.

## 4. Resultset Flow Signaling

Both protocols have messages like `FetchDoneMoreResultsets` and `FetchDoneMoreOutParams`, controlling the flow of multiple resultsets and out-parameter sets.

- **MySQL X Protocol**: Focuses on a linear flow—send columns, send rows, then `FetchDoneMoreResultsets` or `FetchDone` to signal transitions.
- **PolarDB-X**: Maintains the same conceptual flow but adds more data transfer modes (chunks and token control) and provides additional metadata at the end of fetches.

## 5. Use Cases and Intent

**MySQL X Protocol:**

- Designed as a generic protocol for communicating with MySQL’s X Plugin, focusing on compatibility, simplicity, and a straightforward, row-based messaging system.

**PolarDB-X:**

- Enhanced for a distributed, possibly large-scale database scenario where:
  - More efficient data streaming is necessary (chunks, column-based transfer).
  - Additional metadata (original MySQL type codes, examined rows, chosen index) is beneficial.
  - Better integration with distributed systems or cluster management (e.g., tokens) is needed.

## 6. Backward Compatibility and Extensions

The PolarDB-X `Resultset` definition appears as a superset of MySQL X Protocol’s approach, maintaining compatibility for basic fields and messages while adding optional fields and entirely new message types that can be ignored by simpler clients if not utilized. This suggests an evolution of the X Protocol concept to suit more complex database environments.

------

**In Summary**, the core difference is that PolarDB-X extends the MySQL X Protocol’s resultset model by introducing chunk-based, column-oriented data representation, providing richer metadata (including original MySQL type info), and delivering additional execution statistics. Meanwhile, MySQL X Protocol’s `resultset.proto` focuses on a simpler, row-by-row message model suitable for straightforward query execution and retrieval.

# SQL

Below is a thorough comparison between the MySQL X Protocol’s `mysqlx_sql.proto` and the PolarDB-X `polarx_sql.proto` definitions, as well as the additional `polarx_execplan.proto` file that PolarDB-X introduces. These files cover how SQL statements are sent and executed, and the enhancements PolarDB-X introduces over the standard MySQL X Protocol.

## 1. Shared Core Concept

**MySQL X Protocol (mysqlx_sql.proto)**:

- Defines a `StmtExecute` message for sending SQL or namespace-based statements to the server.
- `StmtExecute` includes a statement (`stmt`), optional parameters (`args`), and a boolean `compact_metadata` flag to control metadata verbosity.
- Returns zero or more resultsets followed by a `StmtExecuteOk` message to signal the successful completion of the command.

**PolarDB-X Protocol (polarx_sql.proto)**:

- Also defines a `StmtExecute` message, conceptually similar to MySQL X.
- The server responds with zero or more resultsets followed by `StmtExecuteOk`.
- Maintains compatibility with the essential logic of MySQL X Protocol for executing statements.

Both protocols share the baseline concept: a client sends a `StmtExecute` message containing the SQL statement (or command) and parameters, and the server responds with resultsets and a success message. However, PolarDB-X extends the message with additional fields and introduces entirely new message types in a separate protobuf to handle more complex scenarios.

## 2. Differences in `StmtExecute` Message

**MySQL X Protocol `StmtExecute`:**

```
protobufCopiar códigomessage StmtExecute {
  optional string namespace = 3 [ default = "sql" ];
  required bytes stmt = 1;
  repeated Mysqlx.Datatypes.Any args = 2;
  optional bool compact_metadata = 4 [ default = false ];
}
```

**PolarDB-X `StmtExecute`:**

```
protobufCopiar códigomessage StmtExecute {
  optional string namespace = 3 [default = "sql"];
  optional bytes stmt = 1;
  optional bytes stmt_digest = 12;
  optional bytes hint = 13;
  optional bool chunk_result = 14 [default = false];
  optional bool feed_back = 16 [default = false];
  repeated PolarXRPC.Datatypes.Any args = 2;
  optional bool compact_metadata = 4 [default = false];

  optional string schema_name = 5;
  repeated PolarXRPC.Datatypes.SessionVariable session_variables = 6;
  optional string encoding = 7;
  optional int32 token = 8;
  optional bool reset_error = 9;
  optional uint64 snapshot_seq = 10;
  optional uint64 commit_seq = 11;
  optional bool use_cts_transaction = 15;
  optional uint64 capabilities = 17;
  optional bool mark_distributed = 18;
  optional bool query_via_flashback_area = 19;
}
```

**Key Enhancements in PolarDB-X:**

- **`stmt_digest`**: Possibly used for performance monitoring, caching, or identifying similar queries.
- **`hint`**: Allows providing execution hints to the server.
- **`chunk_result`**: Indicates that results should be returned in a chunked format (as seen in the resultset enhancements), potentially improving performance for large result sets.
- **`feed_back`**: Could signal the server to provide additional execution metrics or hints back to the client.
- **Transaction and Snapshot Fields (`snapshot_seq`, `commit_seq`, `use_cts_transaction`)**: Reflect distributed transaction features, allowing precise control over snapshot and commit order in a distributed environment.
- **Session and Schema Control (`schema_name`, `session_variables`, `encoding`)**: These fields let the client specify schema context, session variables, and character encoding directly in the request.
- **`reset_error`**: Possibly used to reset error states in ongoing transactional workflows.
- **`capabilities`, `mark_distributed`, `query_via_flashback_area`**: Indicate server capabilities, distributed query execution markers, and flashback query capabilities (time-travel queries or historical snapshots).

In short, PolarDB-X augments the execution request with a wide range of options and metadata, reflecting a more sophisticated, distributed, and feature-rich environment than standard MySQL X Protocol.

## 3. Additional Messages

**MySQL X Protocol:**

- Only defines `StmtExecute` and `StmtExecuteOk` in the `mysqlx_sql.proto` file.

**PolarDB-X Protocol:**

- Retains `StmtExecuteOk` as a simple acknowledgment of success.
- Introduces a `TokenOffer` message to handle token-based flow control or security tokens.

Moreover, PolarDB-X provides a completely separate proto file (`polarx_execplan.proto`) that defines a second mechanism for sending queries:

### `polarx_execplan.proto` Additional Capabilities

This file does not exist in MySQL X Protocol. It defines an advanced and entirely different method (`ExecPlan`) for sending queries:

- **`ExecPlan` Message**: Allows sending a pre-built execution plan, not just a raw SQL statement. This suggests the client can transmit a structured execution plan (like a logical plan tree) instead of a simple SQL string.
- Supports various plan types (`GET`, `TABLE_SCAN`, `RANGE_SCAN`, `PROJECT`, `FILTER`, `AGGR`, etc.) and sophisticated constructs, allowing the client to influence or directly specify how the server should execute queries.
- Includes `Transaction`, `AnyPlan`, `GetPlan`, `TableScanPlan`, and other complex messages to define how data should be retrieved, filtered, aggregated, and projected.
- Provides fields for `session_variables`, `reset_error`, `chunk_result`, `feed_back`, and other parameters, much like `StmtExecute`, but aimed at a more "query engine" level of control rather than just sending SQL text.
- The presence of `GetTSO`, `ResultTSO`, and `AutoSp` messages further shows integration with distributed transaction management and snapshot handling.

**Implications:**

- PolarDB-X allows two distinct approaches:
  1. Send standard SQL statements via `StmtExecute` for compatibility and simplicity.
  2. Send a fully structured execution plan (`ExecPlan`) with parameters and metadata, enabling advanced clients or middleware to directly convey optimized, pre-compiled plans.

No equivalent capability exists in the standard MySQL X Protocol, which only deals with sending SQL statements and parameters in a more generic manner.

## 4. Use Cases and Intent

**MySQL X Protocol:**

- Optimized for simple execution of SQL or CRUD operations in a relational or document store.
- Keeps `StmtExecute` minimal: a SQL statement and parameters.

**PolarDB-X:**

- Geared towards complex, distributed scenarios.
- The `StmtExecute` message adds parameters for distributed transactions, snapshot isolation, flashback queries (time-travel), and other advanced features.
- The `ExecPlan` approach allows the client or a middleware layer to push down pre-computed, possibly cost-optimized plans to the server. This could be crucial in large-scale, distributed environments where query optimization is done client-side or via a separate planner.

## 5. Compatibility and Extensions

PolarDB-X’s `StmtExecute` remains compatible with the core MySQL X Protocol approach (you can still just send a SQL statement), but it offers extended fields for advanced use. The introduction of `ExecPlan` and related messages in a separate file does not break compatibility but offers an optional, more advanced interaction model.

**This dual mechanism—plain SQL vs. fully specified exec plans—demonstrates PolarDB-X’s flexibility and advanced feature set aimed at distributed databases or complex query processing scenarios.**

------

**In Summary**:

- **MySQL X Protocol (SQL messages)**: Simple, straightforward—send SQL, get results.
- **PolarDB-X (SQL messages)**: Enhanced `StmtExecute` with additional metadata for distributed transactions, hints, capabilities, and flashback queries.
- **PolarDB-X (ExecPlan messages)**: A second, more powerful option to send queries as structured execution plans rather than just SQL text, enabling fine-grained control over how queries are executed in a distributed environment.