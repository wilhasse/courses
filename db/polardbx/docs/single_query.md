# Overview

PolarDB-X is a distributed relational database that separates the frontend (SQL entrypoint, parser, optimizer, router) from the backend (storage nodes often running a forked version of MySQL for actual data processing). The frontend layer is implemented largely in Java and is responsible for understanding SQL queries, optimizing them, and determining how to route them to the appropriate backend node(s). Once the route is determined and the query execution plan is formed, the frontend sends one or more SQL statements to the backend MySQL forks for execution. The backend nodes handle the low-level operations—like indexing, data retrieval, and transaction semantics—then return results to the frontend, which in turn merges and returns them to the client.

In a simple, non-distributed scenario—where the table or query doesn’t require sharding or federating results from multiple backend nodes—the process is more straightforward. The query essentially goes through parsing and optimization in the frontend, and then is sent as a single (possibly slightly transformed) SQL statement to a single backend MySQL instance.

# Detailed Breakdown Step-by-Step:

Client Connection and Request Handling
The client (e.g., a JDBC application, a command-line tool, or a service) connects to the PolarDB-X endpoint. This endpoint is the Java-based frontend layer.

The client submits a simple SQL query, for example:

```sql
SELECT * FROM employees WHERE id = 123;
```

## SQL Parsing in the Frontend
The PolarDB-X frontend includes a SQL parser implemented in Java. This parser is typically generated from a grammar (often using a parser generator tool or integrated parser libraries). It is responsible for:

Lexing: Breaking the SQL string into tokens like SELECT, *, FROM, employees, WHERE, id, =, 123.
Parsing: Constructing an Abstract Syntax Tree (AST) that represents the query structure. For our simple query, the AST might have a SELECT node, a TABLE node (employees), and a WHERE clause node (id = 123).
Logical Plan Construction
After parsing, the PolarDB-X frontend transforms the AST into a logical execution plan. The logical plan is a more structured representation of what the query wants to do, independent of physical execution details.
For a simple non-distributed query, the logical plan is quite direct:

A single table scan on employees filtered by id = 123.

## Query Optimization

The frontend includes an optimizer (in Java) that applies various rules and transformations. In a distributed scenario, it would consider sharding keys, possible indexes, join strategies, and whether to route to multiple backend nodes. In a simple single-table scenario:

The optimizer might check for available indexes on employees.id.
If there’s an index, it decides to do an indexed lookup rather than a full table scan.
It might simplify or rearrange the predicate if needed.
Because our scenario isn’t distributed (let’s assume just one underlying backend database), the optimizer quickly determines that this query can be directly executed on the single backend node holding the employees table.

## Routing Decision
In the distributed architecture, a key step is routing. The frontend determines which backend node(s) hold the relevant shards of the data. For a non-distributed scenario:

Routing is trivial. There is only one backend node or one database instance. The query is mapped directly to that node.
The routing layer (still part of the Java frontend) knows the topology. It sees that the employees table is stored on a particular backend node and chooses that node’s connection pool for execution.
Physical Plan Generation
Now the PolarDB-X frontend turns the logical plan into a physical execution plan. In a distributed scenario, this might involve generating multiple SQL statements, one per shard, and a merge operation afterward. In our simple scenario, the physical plan is just the direct SQL statement that will be sent to the backend MySQL instance. It might look identical or nearly identical to the original query, for example:

```sql
SELECT * FROM employees WHERE id = 123;
```
Or, if the optimizer added hints or changed the query slightly (for example adding index hints), it would generate something like:


```sql
SELECT /*+ INDEX(employees id_index) */ * FROM employees WHERE id = 123;
```

## Communication with the Backend (MySQL Fork)

With the physical plan ready, the frontend uses its connection layer to send the SQL to the backend. PolarDB-X typically uses a standard MySQL protocol or a modified version of it to communicate with the backend MySQL engine. Under the hood:

The Java frontend maintains a pool of connections to the backend.
It picks a free connection from the pool.
It sends the SQL text over the MySQL protocol.
Since PolarDB-X backend nodes are MySQL forks (often based on Alibaba’s fork of MySQL or PolarDB for MySQL), they understand this protocol and the SQL statement just like a standard MySQL server would.

Query Execution in the Backend
The backend MySQL fork receives the query and executes it against its local storage engine. For a simple SELECT:

The backend checks its indexes.
Locates the row(s) with id = 123.
Reads the data pages from disk or memory.
Constructs the result set to return.
The backend node then sends the result rows back over the MySQL protocol to the frontend.

Result Assembly in the Frontend
In a distributed query scenario, the frontend may need to merge results from multiple shards, apply orderings, or combine partial aggregates. But in our simple scenario (a single query on a single backend node), no merging is needed. The frontend just passes the results directly back to the client.

## The frontend:

Receives the rows from the backend.
Possibly converts internal data formats if needed.
Streams the final result set back to the client using the client’s connection protocol (often the MySQL wire protocol if the client expects that).
Return Results to the Client
Finally, the client gets the result set. From the client’s perspective, it connected to PolarDB-X, issued a query, and got results—just as if it were talking to a regular MySQL database, but behind the scenes, PolarDB-X managed parsing, optimization, and routing.

Key Libraries and Components Involved:

Parser Library: Converts SQL text to an AST. PolarDB-X uses a custom parser implemented in Java, potentially adapted from or inspired by MySQL’s parser grammar.
Optimizer and Planner: A series of Java classes that apply rules to produce an optimized logical and then physical plan. May use frameworks or rule-based engines under the hood.

Routing and Metadata Manager: Keeps track of database topology (schemas, tables, shards) and decides how to map logical tables to physical tables/backends. In the non-distributed scenario, it’s straightforward: one table maps to one backend.
Execution Engine: Manages connections to backend nodes, sends the SQL text, handles result sets, and orchestrates the distributed execution. Even in a non-distributed scenario, it’s the same layer but only has to deal with a single node.
MySQL Protocol Layers: The frontend implements the MySQL wire protocol in Java to handle client connections and backend connections. This includes handling authentication, handshake, query command packets, and result set packets.

## Summary:

In a simple (non-distributed) scenario, PolarDB-X’s Java-based frontend still follows the same general flow—parse, optimize, route, execute—but without the complexity of multi-shard routing or result merging. The result is a direct pipeline from a client’s SQL query through a Java-based parser and optimizer, resulting in a single SQL statement sent to a single MySQL-like backend for execution, and a straightforward return of the resulting rows to the client.