Below is a high-level overview of how PolarDB-X (the proprietary/distributed version by Alibaba) processes SQL queries internally and how it might leverage an internal execution plan (often called **ExecPlan**, **PhysicalPlan**, or simply **execution plan**) to distribute requests via XProtocol. Although some details are specific to Alibaba’s internal architecture, the general concepts may help you if you want to reuse (or mimic) some of PolarDB-X’s logic on a single-node Percona XProtocol setup.

------

## 1. Big Picture of PolarDB-X Query Flow

1. **SQL Parsing & Validation**
   - **Parser**: Converts SQL text into an Abstract Syntax Tree (AST).
   - **Validator**: Checks syntax correctness, table/column existence, privileges, etc.
2. **Logical Plan Construction**
   - The AST is converted into a **logical plan** (often using a framework such as Apache Calcite).
   - Logical plan is a tree of relational operators (e.g., `LogicalProject`, `LogicalJoin`, etc.), reflecting the *intended* semantics of the query without yet mapping to physical tables, indexes, or distribution details.
3. **Optimizer (CBO & Rule-based)**
   - The logical plan is optimized. This typically involves:
     - **Rule-based transformations** (pushdown of predicates, merging projections, etc.).
     - **Cost-based optimization** (estimating row counts, choosing indexes, join orders, etc.).
   - The result is still (logically) a single plan, but with transformations that reflect the best approach to retrieve data *in a distributed environment*.
4. **Physical Plan / ExecPlan Generation**
   - The logical plan is translated into a **physical plan**. In a distributed system, the physical plan is often broken down by **shards** or **storage nodes**.
   - Each step that references one or more base tables is assigned to a particular storage node (or multiple, if broadcast or partitioned).
   - In PolarDB-X, each final piece of the plan that goes to a single node becomes (roughly) a separate “sub-plan,” “task,” or “table scan request,” often routed over XProtocol.
5. **Plan Execution**
   - The coordinator node (the PolarDB-X server node) takes the physical plan and coordinates the execution:
     1. **Sends sub-requests** over XProtocol to each storage node or backend.
     2. **Receives partial results**, does necessary merges or aggregates.
     3. Returns final result to the client via the MySQL protocol (or XProtocol if the client is speaking that).

------

## 2. Where the “ExecPlan” Fits In

In the PolarDB-X codebase, you’ll see references to plan objects like:

- `RelNode` (Calcite-based)
- `LogicalView`, `LogicalJoin`, `LogicalSort`, etc. (logical operators in TDDL/PolarDB-X)
- `PhyTableScan`, `PhyHashJoin`, `PhyProject`, etc. (physical operators)
- `ExecPlan` or `ExecutionPlan` often is a container of these physical operators or sub-plans.

Essentially:

1. **ExecPlan** = the container describing how to run the query physically.
2. It contains multiple “plan fragments” or “sub-plans,” each of which might map to a single storage node or a single request.

When you see references to “table-by-table data request,” that typically means **each physical operator** that needs data from a storage node is turned into an **XProtocol request**. For instance:

```
text


Copiar código
SELECT * FROM tableA WHERE colA = 123
```

- If `tableA` is physically located on a single node, you might have just one `PhyTableScan` plan node, which is turned into a single XProtocol `COM_QUERY` or “execute plan” message.
- If `tableA` is partitioned across 4 nodes, you might create 4 sub-requests, each scanning the relevant partition.

PolarDB-X then merges partial results (like a “coordinator aggregator”).

------

## 3. How PolarDB-X Uses XProtocol Internally

PolarDB-X can connect to MySQL-compatible backends (including Percona, AliSQL, etc.) via:

- **Traditional MySQL Protocol** (COM_QUERY, COM_STMT_EXECUTE, etc.)
- **XProtocol** (more modern, row-based message streaming).

There are classes (in `com.alibaba.polardbx.rpc` or similar) that handle the creation of an `XConnection`, sending raw XProtocol messages (or a helper around them), receiving result-stream events, etc. This is the same portion of the code you’re looking at in your example: it uses an `XConnectionManager` and an `XConnection` object to do `execQuery()` or `execUpdate()`.

Under the hood, there’s a flow somewhat like:

1. Generate a statement / plan in a protobuf-based representation (for XProtocol).
2. Send the plan to the node(s) over XProtocol streams.
3. Read streaming results as protobuf messages.
4. Convert them into row objects, or produce them directly to the client.

------

## 4. Single-Node Scenario vs. Distributed

Since you mentioned your environment is *single-node*, the flow is simpler:

- You may not need any advanced “sharding” logic or multi-phase aggregator.
- You parse -> build a plan -> realize that there’s only one node to connect to.
- You generate a single sub-plan that does the entire query on that node.
- You still send that sub-plan over XProtocol (like a normal MySQL XProtocol request), then collect rows.

If you’d like to break queries more granularly (for caching or partial optimization), you can do so in your **physical plan** generator. For example, if you see `JOIN tableA, tableB`, you could decide to:

1. Send a partial request to get columns from `tableA` (and maybe cache them).
2. Send another partial request for `tableB`.
3. Do the join locally.
4. Return final results to client.

That is basically how a distributed system tries to push down or partially push down queries. But you can do it on a single node if you want finer control, or if you’re trying to build your own caching/acceleration layer.

------

## 5. Key Modules in the PolarDB-X “SQL Layer”

Although PolarDB-X is partially proprietary, some open-source code or public references show roughly these modules:

1. **TDDL (Taobao Distributed Data Layer)** or “GalaxySQL” – The core query engine and optimizer that originated from Alibaba.
2. **Planner/Optimizer** – Based on Apache Calcite or a fork. The code takes the SQL AST, turns it into a Calcite `RelNode` graph, then transforms it to a distributed plan.
3. **Executor** – The runtime that interprets the final plan:
   - For each physical node (or partition), generate the actual SQL or XProtocol request.
   - Dispatch that request.
   - Gather results and combine them if needed (for sorting, aggregation, join, etc.).
   - Streams them back to the client.
4. **RPC / Net** – This is where you see the `XConnection`, `XConnectionManager`, `execQuery()`, `execUpdate()`, etc. The code encodes queries into XProtocol messages and decodes the results.

------

## 6. Adapting PolarDB-X Logic for a Single-Node “Proxy”

If your goal is to:

1. Use a PolarDB-X-like SQL parsing/optimizer pipeline (maybe from TDDL or a Calcite-based approach).
2. Then intercept the final plan (the **ExecPlan**) and decide how you want to do chunked or partial reads from the single backend.
3. Possibly implement caching or rewrite the plan.

You can follow these steps:

1. **AST / Logical Plan**
   - Reuse or replicate the parser from TDDL or MySQL parser.
   - Build the intermediate representation.
2. **Rewrite**
   - Insert your logic to “intercept” certain queries or do partial caching.
3. **Physical Plan**
   - If your environment is truly single-node, the “distributed” logic might be overkill.
   - But you can still break the plan into smaller sub-steps if you want partial execution or better local caching.
4. **Execution Over XProtocol**
   - Construct XProtocol messages for each sub-plan.
   - Send them to your Percona XProtocol node.
   - If you need merges (e.g., local join), do so.

Because your environment differs from the official PolarDB-X (in that you do not have multiple shards/nodes), a lot of the distributed plan logic might be unnecessary. But you can still glean how they:

- Handle query pushdown,
- Generate the actual SQL or proto message for each sub-plan,
- Stream the results,
- Return them in MySQL wire protocol to the client (if your client is using MySQL protocol).

------

## Summary

- **PolarDB-X** uses a fairly standard “parse -> logical plan -> optimize -> physical plan -> distributed execution” flow, with “ExecPlan” being the final representation that says “go fetch data from these physical shards.”
- Each shard request can be turned into one or more **XProtocol** messages (like `execQuery()` calls).
- Results are streamed back to the coordinator, combined if needed, and returned to the client.
- For a single-node scenario, the same flow is conceptually simpler—**no real distribution**—but you can still break queries into sub-requests for local caching or partial optimization.

Hopefully this clarifies how PolarDB-X uses an **ExecPlan** to drive table-by-table (or shard-by-shard) requests via XProtocol. Even though you’re only targeting a single Percona XProtocol node, you can still adopt the same architecture if you plan to do local caching or custom partial processing.