Below is a **conceptual map** showing how you can **connect** the **steps** of the PolarDB-X query flow (parsing, logical plan, physical plan, etc.) to the **source code classes** in the PolarDB-X repository. Because the codebase is **large** and **complex**, I'll focus on the **key classes** that handle each phase. You can then navigate from there into more details as you explore the code.

------

## 1. Parsing / SQL to AST

- **Key Concept**: Turn the raw SQL string into an AST (Abstract Syntax Tree).

- Where to Look:

  1. `com.alibaba.polardbx.optimizer.parse.FastsqlParser`

      and 

     `FastsqlUtils`

     - Uses the Druid / FastSQL parser to parse text into an AST or `SqlNode`.

  2. `com.alibaba.polardbx.optimizer.parse.visitor.CalciteVisitor`

     - Converts the Druid AST into Calcite `SqlNode`.

Most of the parser code is inside **Druid** plus **FastSQL** integration. You’ll see:

- **`MySQLLexer`**, **`ServerParse`** if you look at MySQL syntax recognition for simple commands.
- **`HintParser`**, **`TDDLSqlFunction`**, etc., if you’re exploring extended hints or DDL syntax.

------

## 2. Building a **Logical Plan** (Calcite `RelNode` Tree)

- **Key Concept**: Translate the parsed SQL AST into a **Calcite** relational operator tree (e.g. `LogicalFilter`, `LogicalProject`, etc.).

- Where to Look:

  1. `com.alibaba.polardbx.optimizer.core.planner.SqlConverter`

     - Entry point that takes `SqlNode` and does `SqlToRelConverter` calls, producing a `RelNode` tree.

  2. `org.apache.calcite.sql2rel.SqlToRelConverter`

      (Calcite)

     - The underlying Calcite class.

  3. `com.alibaba.polardbx.optimizer.core.rel.LogicalView`

     - This is where queries referencing actual tables end up as a `LogicalView` node.

  4. `com.alibaba.polardbx.optimizer.core.rel.*`

      classes

     - E.g. `LogicalFilter`, `LogicalProject`, `LogicalJoin`, `LogicalModify`, which are custom “logical” operators in PolarDB-X.

------

## 3. Logical Optimization

- **Key Concept**: Rewrite or simplify the logical plan. (Push filters down, merge projects, rewrite subqueries, etc.)

- Where to Look:

  1. `com.alibaba.polardbx.optimizer.core.planner.Planner`

     - The main “entry” for planning.

  2. `com.alibaba.polardbx.optimizer.core.planner.rule.*`

     - Contains Calcite 

       Rules

        or “HepPlanner” transformations, e.g.

       - **`FilterProjectTransposeRule`**, **`FilterMergeRule`**, **`ProjectMergeRule`**, etc.

These transformations remain in the **logical** domain, but they rearrange the node tree for efficiency.

------

## 4. Physical Plan Generation (where distribution / shard logic is decided)

- **Key Concept**: Takes the optimized `RelNode` (logical) and decides exactly how to execute: which shard (if distributed), which index, how to parallelize, etc.

- Where to Look:

  1. `com.alibaba.polardbx.optimizer.core.rel.PushDownOpt`
     - **This** is where the system “pushes down” logical operators into a physical plan (especially for `LogicalView`).
     - It also does partition pruning, sets up sub-plans for each shard or group.
  2. `com.alibaba.polardbx.optimizer.core.rel.PhyTableOperation`
     - The final “physical” scan or DML node referencing the real table + SQL.

In **PolarDB-X**, the code that decides which shards to read is mostly in:

- **`com.alibaba.polardbx.optimizer.partition.pruning.PartitionPruner`**
- **`com.alibaba.polardbx.optimizer.partition.pruning.PartPruneStep`**
- **`com.alibaba.polardbx.optimizer.core.rel.PushDownOpt`** (especially `buildTargetTables` or `getRelShardInfo`).

------

## 5. Execution & Merging

- **Key Concept**: After the physical plan is built (the engine knows “which sub-SQLs to run on which shards”), the system **executes** them, merges partial results, and returns a final result set to the user.

- Where to Look:

  1. `com.alibaba.polardbx.executor`

      package

     - For example, **`com.alibaba.polardbx.executor.operator`** has operator classes like `HashJoinExec`, `SortExec`, etc., which run “locally” on partial results.
     - **`com.alibaba.polardbx.executor.operator.scan`** or **`TableScanClient`** for scanning.

  2. `com.alibaba.polardbx.executor.common.ExecutorContext`

     - Coordinates the overall execution environment.

  3. `com.alibaba.polardbx.executor.mpp`

     - If MPP or parallel execution is used, scheduling or parallel merges are handled here.

  4. `com.alibaba.polardbx.executor.spi.IRepositoryFactory`

      / 

     `PlanExecutor`

     - The final step is often calling **`PlanExecutor.execute(...)`** with a fully prepared physical plan.

In a single-shard scenario, it might just run one SQL on one node. In a multi-shard scenario, it might create multiple **sub-queries** in parallel, then aggregate them.

------

## 6. Putting It All Together (An Example Path in the Code)

1. **SQL enters** via a MySQL front-end connection in **`com.alibaba.polardbx.net.handler.QueryHandler`** or **`com.alibaba.polardbx.server.handler.\*`**.
2. The string is **parsed** by **`com.alibaba.polardbx.optimizer.parse.FastsqlParser`** → yields a **`SqlNode`**.
3. It is converted to a **logical plan** via **`com.alibaba.polardbx.optimizer.core.planner.SqlConverter`** → yields a Calcite **`RelNode`** tree.
4. The **logical plan** is **optimized** by various **HepPlanner** rules in **`com.alibaba.polardbx.optimizer.core.planner.rule.\*`**.
5. The system calls **`LogicalView.optimize()`, `PushDownOpt.optimize()`,** etc. to produce a **physical plan** (sub-queries for each shard, or a single sub-plan for a single shard).
6. **Execution**: The final plan is run by **`com.alibaba.polardbx.executor.ExecutorHelper`** or **`com.alibaba.polardbx.executor.spi.IRepositoryFactory`** → eventually sub-queries are issued to the backend (MySQL or PolarDB-X shard).
7. **Results** are merged if multi-shard; partial results might be handled by **`com.alibaba.polardbx.executor.operator.\*`**.
8. The final row set is returned to the client.

------

## 7. References & Classes to Focus On

1. Parsing

   - `FastsqlParser`, `FastsqlUtils`, `Druid` classes.

2. Building the Calcite `RelNode`

   - `SqlConverter`, `SqlToRelConverter`.

3. `LogicalView`

   - Key class representing a single “logical table,” yet can contain joins, filters, etc.

4. `PushDownOpt`

   - The engine for rewriting that “logical” sub-tree into final “physical” sub-queries.

5. Partitioning / Sharding

   - `PartitionPruner`, `PartitionPruneStep`, `PartitionPrunerUtils`.

6. Execution

   - In 

     ```
     com.alibaba.polardbx.executor
     ```
     :

     - `operator` (various operators)
     - `PlanExecutor` (entry point for running a plan)
     - `ExecutorHelper`

7. `PlanManager` / `PlanCache`

   - If you want to see how plans are cached or reused.

These modules orchestrate the entire journey from SQL text → AST → logical plan → physical plan → execution → results.

------

## 8. Learning Strategy

- **Start** with simpler classes like `LogicalView` and `PushDownOpt` to see how a single table’s plan is built and pushed.
- **Check** the **Calcite** integration in `SqlConverter` or `Planner` to see how the engine transforms `SqlNode` → `RelNode`.
- **Explore** the `executor` package to see how sub-plans are executed or merged.
- **Search** for tests or examples in the code (like `Planner4Test` or `PushDownOptTest`) to see real usage.

By understanding these critical classes, you’ll have a good mental map of how the plan-flow steps align with PolarDB-X’s code.