# Logical View

Source:   
polardbx-optimizer/src/main/java/com/alibaba/polardbx/optimizer/core/rel/LogicalView.java

Below is a **detailed explanation** of what **`LogicalView`** does in the PolarDB-X codebase and **how you might use it** (or adapt its ideas) when building your own single-node or distributed SQL engine. The class is quite large and packed with functionality, so we’ll break it down step by step.

------

## 1. What **`LogicalView`** Is in General

- **`LogicalView`** extends Calcite’s `TableScan` node:
  In most relational optimizers (including Calcite), a `TableScan` means “we are reading from some underlying table.” `LogicalView` is a **special** kind of `TableScan` that represents a **logical query** on top of possibly **multiple** physical partitions (or shards) of one or more tables.
- In PolarDB-X:
  1. A **logical table** may be physically stored across multiple shards (multiple MySQL instances), or it may be a single-table or broadcast table.
  2. **`LogicalView`** acts as the single “logical” operator that encloses the query (SELECT / JOIN / FILTER / PROJECT / etc.) that will ultimately be pushed down to the underlying engine(s).
  3. It orchestrates how to **prune** partitions, generate **physical operations** for each shard, manage **pushdown** of predicates, merges subqueries (apply/correlate), or handle locking (e.g., `SELECT ... FOR UPDATE`).
  4. Because it’s a top-level “table-scan”-like node, it also holds references to table names, partition metadata, hints, sub-query rewrite logic, etc.

Essentially, **`LogicalView`** is the primary representation (inside the logical plan) of “how we will read from a distributed (or single) table.” Then, during the optimization phase, it prunes shards, rewrites queries, merges hints, etc., turning itself into a final set of “physical” table scans or “PhyTableOperation” nodes.

------

## 2. Major Responsibilities and Components

### 2.1 **Core Fields**

1. **`dbType`**
   - Usually `MYSQL` or `DrdsType` in PolarDB-X. Tells the system what dialect or pushdown logic to generate (e.g., MySQL dialect).
2. **`tableNames`**
   - A list of logical table names this `LogicalView` includes. In single-table queries, it’s typically size 1; in certain join pushdown scenarios, it may hold multiple table names.
3. **`pushDownOpt`** (type `PushDownOpt`)
   - A helper/manager that contains the **pushed-down** relational tree.
   - For example, if the original user query was `SELECT ... FROM tableA`, after certain transformations, `pushDownOpt` might store a Calcite RelNode tree with filters, projections, sorts, etc., that is “ready” to be turned into a physical plan.
   - It also has logic to handle partial rewriting, pruning, or flattening join conditions.
4. **`lockMode`**
   - Whether the query is a “FOR UPDATE” lock or not. Also relevant for read-committed or read-consistency logic.
5. **Partition-related fields**
   - **`newPartDbTbl`** indicates if the target table(s) are “new partition” or “legacy.”
   - **`pruneStepHintCache`**, **`comparativeHintCache`**, etc., store partition hints or precomputed partition steps.
   - This is how PolarDB-X can prune partitions (only read from the relevant shard(s)).
6. **Sub-query management**
   - **`scalarList`** / **`correlateVariableScalar`** store references to subqueries or correlated variables that appear in the logical plan.
   - The code can rewrite or “unwrap” them if conditions are met (like single-shard pushdown).
7. **XPlan**
   - An internal representation that can generate a “protobuf-based” plan sent via XProtocol to the underlying node. This is a more advanced pushdown mechanism used inside Alibaba’s engine.
8. **Caching**
   - The code tries to build an internal “SQL template” (an AST or partial string) which can be reused for repeated queries, so we don’t reconstruct the same pushdown SQL each time if the structure is the same.

### 2.2 **Key Methods**

1. **`optimize()`** / **`optimizePhySql()`**
   - Invoked when the system wants to finalize the sub-plan, do cost-based or rule-based rewrites, or generate pushdown queries.
2. **`getInput(...)`**
   - Responsible for computing the final “physical” shards and building **`PhyTableOperation`** nodes (the final physical scans or writes) from the `LogicalView`.
   - This is how the system eventually obtains an actual list of “(group, physical_table)” pairs to run, each with its own SQL text.
3. **`getSqlTemplate(...)`**
   - Builds or fetches a cached “SQL AST” (a `SqlSelect`) that references `?` placeholders for parameters and `?` for table placeholders.
   - Later replaced by the actual table name(s) at execution time.
   - This is crucial for the “lazy” or “dynamic” rewriting that picks which shards to query.
4. **`buildTargetTables(...)`**
   - Figures out how many shards or partitions are relevant, given the partitioning rules and the query’s predicates.
   - If it’s a single-shard scenario, it will produce exactly 1 “(group, physical_table)”, otherwise multiple.
5. **`computeSelfCost(...)`** / **`estimateRowCount(...)`**
   - Standard Calcite methods for cost estimation.
   - `LogicalView` includes logic to guess how many rows it might read based on partition pruning and index usage.
6. **Join pushdown logic**
   - Methods like **`pushJoin(...)`**, **`pushSemiJoin(...)`**.
   - If the system detects that we can push down a join of two tables to the same shard, it merges them into a single `LogicalView`.
7. **Partition pruning**
   - **`PartitionPruner.prunePartitions(...)`** and storing partial results in **`PartPrunedResult`**.
   - The code has logic like “If user wrote `TABLE t PARTITION (p0, p1)`, we filter out everything but `p0, p1`.”
   - For range/hash partitioning, it also dynamically prunes based on the WHERE clause.
8. **Apply / subquery rewriting**
   - The code in `buildApply()` walks the pushedRelNode to find subqueries, tries to rewrite them into standard “RelNodes” if pushdown is possible, etc.

### 2.3 **How the Class Fits in the Query Flow**

1. **User writes `SELECT ... FROM table ... WHERE ...`.**
2. The planner (using Calcite) first constructs a logical plan with `LogicalView` for the table reference.
3. The system calls **`optimize()`** or **`getInput(...)`** on that `LogicalView`.
4. `LogicalView` identifies which shards / partitions apply, merges subqueries if possible, and creates final “physical scans” (`PhyTableOperation`).
5. The executor runs those physical scans on each shard and merges results if needed.

------

## 3. How You Might **Use** or **Adapt** It

Since your goal is to build a single-node engine (Percona XProtocol) but still want to leverage the logic from PolarDB-X:

1. **Parsing & Planning**
   - You could still reuse the part of `LogicalView` that handles rewriting subqueries and building a final “SQL template.”
   - If you do not need multi-shard or multi-node logic, you can simplify or remove partition/shard steps.
   - Instead of “multiple shards,” you might have a single backend, but you can keep the logic that “pushes down” filters, projections, or subqueries.
2. **Subquery / correlation rewriting**
   - Many of the methods in `LogicalView` (and `PushDownOpt`) handle rewriting correlated subqueries or “apply” subqueries into simpler forms.
   - This can be reused even in a single-node system if you want to push subqueries into the server rather than evaluate them in memory.
3. **SQL Template building**
   - The code that constructs a “SQL template” with placeholders, then merges actual parameter values at runtime can be reused for caching.
   - If you call **`getSqlTemplate()`** or **`buildSqlTemplate()`**, you get a `SqlSelect` with question marks in place of parameters. Then at runtime, you just fill in the parameters.
   - This is useful if you have a high QPS system that repeatedly runs the same shape of query with different parameter values.
4. **Costing & metadata**
   - The self-cost logic (`computeSelfCost`) might be partially relevant if you do cost-based optimizations.
   - The method uses row estimates, partition counts, etc.
5. **Partition logic**
   - If you truly do not have multiple partitions, you can skip or stub out calls like `PartitionPruner.prunePartitions(...)`.
   - Or if you eventually do “horizontal partitioning,” the code already has a template for how to decide which partitions to read based on the user’s WHERE clause.
6. **Pushdown**
   - The heart of `LogicalView` is about deciding how to push the operator tree (JOIN, FILTER, AGGREGATE, etc.) down to the storage node. You can see logic like `pushDownOpt.pushJoin(...)`.
   - In a single-node scenario, “pushdown” might be simpler, but you can keep the idea that “filter conditions on the same table” can be combined into a single query.

------

## 4. Important Sub-Classes and Utilities

Within **`LogicalView`**, you’ll see:

- **`PushDownOpt`**: Holds the “pushed” relational tree. A lot of rewriting happens there.
- **`RelToXPlanConverter`**, **`RelXPlanOptimizer`**: Converting from Calcite’s `RelNode` to PolarDB-X’s internal “XPlan” (protobuf) for advanced pushdown. Might not be needed if you only speak MySQL XProtocol.
- **`PartitionPruner`**, **`PartPrunedResult`**: Used heavily for multi-part partition pruning.
- **`PhyTableScanBuilder`**: Creates final physical “table operations” from the pruned shards/tables.
- **`ReplaceTableNameWithQuestionMarkVisitor`**: Rewrites the Calcite AST to place “?” placeholders for table or parameters (for dynamic SQL generation).

------

## 5. Summary & Takeaways

- **`LogicalView`** is a central piece of the PolarDB-X optimizer that represents a logical scan (or multi-scan) over possibly partitioned tables.

- It merges or rewrites subqueries, handles pushdown logic, caches generated SQL templates, prunes partitions, and computes cost-based stats.

- In a 

  single-node

   scenario, you can 

  reuse

   the logic for:

  - Subquery rewriting / correlation unwrapping.
  - Building SQL templates with placeholders.
  - Possibly partial pushdown, if you want to let the backend do the heavy lifting.
  - Cost-based logic if needed.

- You can **remove** or **stub out** the more complex partition / multi-shard features if you only have one Percona node.

If you want a **lighter approach**:

- Look at **`PushDownOpt`** and how it rewrites relational operators.
- Look at **`getSqlTemplate(...)`** for how it builds parameterized SQL.
- Consider how `LogicalView` merges conditions or subqueries, then adapt it to your environment.

This code is quite deep because PolarDB-X is a **full distributed** system with advanced pushdown. You’ll likely keep the aspects that matter for your single-node usage and drop the multi-shard partitioning if it’s not needed.