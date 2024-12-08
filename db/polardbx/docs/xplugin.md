# Introduction

How the PolarX RPC (Remote Procedure Call) protocol operates as a communication layer between a front-end SQL interface (often a proxy, gateway, or a client-side component) and a PolarDB-X MySQL-compatible backend. The protocol is specified using Protocol Buffers (protobuf) and is conceptually similar to the MySQL X Plugin protocol, with its own extensions to meet the distributed SQL engine’s requirements.

# Files

- polarx.proto
- polarx_connection.proto
- polarx_datatypes.proto
- polarx_exec_plan.proto
- polarx_expect.proto
- polarx_expr.proto
- polarx_notice.proto
- polarx_physical_backfill.proto
- polarx_resultset.proto
- polarx_session.proto
- polarx_sql.proto

# High-Level Overview

## Context
PolarDB-X is a distributed SQL engine that can scale out MySQL storage across multiple nodes. A front-end SQL layer (such as a router or a dedicated proxy) needs a standardized, efficient, and rich protocol to communicate queries, sessions, authentication data, and capabilities to the backend MySQL nodes. Instead of using the traditional MySQL protocol, which is procedural and less extensible, PolarX RPC employs a message-based binary protocol defined via protobufs.

## Goals of the Protocol

To allow the front SQL layer to send SQL queries, execute plans, and manage sessions on the distributed backend.
To provide a structured message format that can handle capabilities negotiation, authentication, error handling, execution of SQL statements, retrieval of result sets, and streaming of rows.

To be extensible, allowing for easy addition of new message types (e.g., for distributed transactions, plan execution, file operations, or special commands).
Message Exchange Basics:

All communication happens through sending and receiving protobuf-encoded messages over a bidirectional channel (e.g., a TCP socket). Each message is identified by a type code (an enum) and is followed by the message payload defined in the .proto files. The client (front-end) sends a request message, and the server (PolarDB-X MySQL node) responds with one or more response messages.

# Key Components of the Protocol

## CapabilitiesGet / CapabilitiesSet
Before running queries, the front-end can discover what the backend supports and set certain options atomically. For example, the front-end might request supported features (chunked result sets, compression, etc.) via CON_CAPABILITIES_GET. The backend responds with CONN_CAPABILITIES containing a list of Capability messages. Then the client might use CON_CAPABILITIES_SET to enable or disable certain features.

Connection Close:
The front-end can send CON_CLOSE when it wants to tear down the connection. The server responds with Ok upon completion.
Session Management and Authentication:

## Session Authentication
The front-end initiates authentication with SESS_AUTHENTICATE_START. This includes sending the authentication mechanism name and initial credentials. The server may respond with SESS_AUTHENTICATE_CONTINUE multiple times if more authentication data exchange is needed. Once the server is satisfied, it sends SESS_AUTHENTICATE_OK.  

Session Reset / Close:
The front-end can send SESS_RESET to reset the current session’s state without tearing down the connection. SESS_CLOSE ends the session. The server responds with Ok messages.
Executing SQL Statements:

## StmtExecute

This is the primary message the front-end sends to run an SQL statement. The StmtExecute message includes
namespace: Typically "sql", indicating standard SQL execution.  
stmt: The SQL statement itself as bytes (for example, SELECT ...).  
args: Bind parameters if the statement contains placeholders.  
compact_metadata: A boolean to request minimal column metadata if desired.  
Once the server executes the statement, the response can include:

## Result Set Messages

The server may respond with a sequence of messages representing the result  
RESULTSET_COLUMN_META_DATA: Describes the column types and attributes.  
RESULTSET_ROW or the newer chunked result format with Chunk and Column messages: Contains actual row data.  
RESULTSET_FETCH_DONE or RESULTSET_FETCH_DONE_MORE_RESULTSETS: Signals the end of the current result set and possibly the availability of more results.  
StmtExecuteOk: After all result sets are sent, the server sends SQL_STMT_EXECUTE_OK to indicate successful completion.  
On errors, the server sends an ERROR message detailing the SQL state, code, and message.  

## CRUD Messages
Although not exclusively used for standard SQL, the protocol supports messages like CRUD_FIND, CRUD_INSERT, CRUD_UPDATE, CRUD_DELETE for direct document-store style operations. In a standard SQL scenario, these might not be frequently used, but they show the protocol’s extensibility.  
ExecPlan and Query Plans**: For more advanced features, the protocol defines messages such as EXEC_PLAN_READ or EXEC_SQL and ExecPlan messages that encode a query plan. This allows the front-end to send a pre-compiled or pre-analyzed plan structure to the server, indicating which tables, indexes, filters, and projections are involved.  
ExecPlan messages contain AnyPlan structures describing various plan types (GET, TABLE_SCAN, PROJECT, FILTER, AGGR, etc.), dynamic parameters, session variables, and execution capabilities.  
The server executes this plan and returns result sets similarly to how it returns them for StmtExecute.    
Expect Blocks (Condition Checking): The protocol includes EXPECT_OPEN and EXPECT_CLOSE messages to define conditional execution blocks. The front-end can specify conditions (like no_error, gtid_executed_contains) that must be met for subsequent commands to run. If conditions fail, the server aborts the enclosed commands.

File Operations: Specialized messages for file operations (like FILE_OPERATION_GET_FILE_INFO, FILE_OPERATION_TRANSFER_FILE_DATA) are defined, likely for advanced tasks such as DDL operations, data import/export, or backfill operations in distributed storage.

These messages allow the front-end to request file metadata, transfer file chunks, or manage files directly on the backend nodes, which is critical for some distributed maintenance tasks.
Notices and Warnings: The server can send asynchronous notifications back to the client:

NOTICE: Encapsulates WARNING, SESSION_VARIABLE_CHANGED, SESSION_STATE_CHANGED messages.
Warnings and state changes keep the front-end aware of things like schema changes, session variable modifications, or transaction state without breaking the primary request/response flow.
Error and OK Messages:

ERROR: Describes issues with severity, code, SQL state, and message.  
OK: A generic success response for operations like CapabilitiesSet, Close, Reset, etc.
Communication Flow Example

## Connection Initialization

Client sends CON_CAPABILITIES_GET.  
Server responds with CONN_CAPABILITIES.  
Client sends SESS_AUTHENTICATE_START with authentication data.  
Server responds with SESS_AUTHENTICATE_OK after successful authentication.  

Running a Query:
Client sends SQL_STMT_EXECUTE with stmt = "SELECT * FROM employees".  
Server may send RESULTSET_COLUMN_META_DATA, followed by multiple RESULTSET_ROW messages or a Chunk-based message set.  
When done, the server sends RESULTSET_FETCH_DONE and then SQL_STMT_EXECUTE_OK.  

Session and Connection End:  
Client sends SESS_CLOSE to end the session.  
Server responds with Ok.  
Client finally sends CON_CLOSE to close the connection.  
Server responds with Ok.  

# Comparison with MySQL X Plugin

## Similarities

Both use Protobuf for structured, extensible message formats.
Both separate authentication, capability negotiation, and statement execution into distinct message types.
Both can stream result sets in a non-blocking fashion.

## Differences

PolarX RPC includes messages and fields tailored to distributed execution, backfill operations, and advanced conditions (like EXPECT blocks).  
PolarX RPC may have specific transaction and snapshot sequence fields (snapshot_seq, commit_seq) to manage distributed transaction states.  
PolarX RPC includes more granular control over execution plans (ExecPlan) and allows direct plan-based execution rather than just text-based SQL.  

## Conclusion

The PolarX RPC protocol defines a rich, message-based interface that the front-end SQL layer can use to interact efficiently with the distributed MySQL backend. It covers the entire lifecycle of a connection, from capabilities and authentication to statement execution, result streaming, and session management. Its similarity to MySQL X makes it familiar in structure, but it is extended with domain-specific features needed for distributed SQL execution and maintenance in PolarDB-X.