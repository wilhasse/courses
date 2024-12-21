# Push Down Optimization

Source:  
polardbx-optimizer/src/main/java/com/alibaba/polardbx/optimizer/core/rel/PushDownOpt.java

## 1. Overall Role of `PushDownOpt`

Recall that **`LogicalView`** is a Calcite `TableScan`-like node that can encapsulate a complex query tree (filters, projections, aggregates, joins, etc.) to be “pushed down” to a single shard or multiple shards. **`PushDownOpt`** is the **core class** inside `LogicalView` that:

1. Maintains the “pushed-down” **RelNode** tree (the subplan that will eventually become physical SQL).
2. Manages rewriting or “optimizing” that subplan so it can be directly converted into SQL (or XPlan).
3. Builds (and caches) partition/Comparative/PruneStep info used for shard or partition pruning.
4. Tracks the **“plain rows”** representation that helps keep track of how each column references an original table/column—critical for correct pushdown logic.

Essentially, **`PushDownOpt`** performs the transformations and caching needed so that a `LogicalView` can quickly figure out:

- How to produce the final “native SQL” string (i.e., the query text that hits the backend).
- Which partitions or shards need to be read from (based on partition pruning or rule-based sharding).
- How subqueries, correlated references, or semi-joins are transformed into a single pushdown query if possible.

------

## 2. Key Responsibilities and Internal Structure

### 2.1 Core Fields

1. **`tableScan`**
   - Reference back to the parent `LogicalView`.
   - Lets `PushDownOpt` read or update info such as table names, indexes, and partition flags.
2. **`builder`** (`RelBuilder`)
   - A Calcite `RelBuilder` that holds the “pushed RelNode”.
   - Methods like `builder.push(rel)`, `builder.join(...)`, `builder.filter(...)` manipulate the underlying relational tree.
3. **`comparatives`** & **`fullComparatives`** (Lists of Maps)
   - Store **Comparative** objects for partitioning or sharding columns.
   - For example, if the user’s query has `WHERE colA = ?`, PolarDB-X might store that in a `Comparative` structure for partition pruning.
4. **`allPartPruneSteps`** (and **`allColumnarPartPruneSteps`**)
   - For **new partition** tables, `PushDownOpt` needs to build a partition prune plan (`PartitionPruneStep`) that identifies which physical partitions are relevant.
   - This is stored/cached by `PushDownOpt` so we don’t re-derive them repeatedly for the same parameters.
5. **`nativeSqlNodeHintCache`**
   - Caches an AST version of the “native” SQL for this subplan. If set, the system might skip rebuilding the same AST every time.
6. **`plainRows`** (`PushDownOpt.PlainRows`)
   - Maintains a “plain row type” plus a mapping from each output column to the **original** table/column index.
   - Ensures that even if Calcite renames columns internally (like when self-joining the same table), we still know which column belongs to which table index.
7. **`shardRelatedInTypeParamIndexes`**
   - A set of parameter indexes where `IN` expressions (like `col IN (?)`) are relevant for dynamic pruning.
   - Helps “dynamic” or “lazy” partition pruning at execution time.

### 2.2 Major Methods

1. **`push(...)`**

   - Called by `LogicalView.push(RelNode)`.
   - If the RelNode is a `LogicalProject`, a `LogicalFilter`, a `LogicalAggregate`, or a `LogicalSort`, it delegates to helper utilities (`pushProject`, `pushFilter`, `pushAgg`, `pushSort`) that manipulate the `builder`.
   - Also updates the `plainRows` mapping to reflect how the output columns change after each pushdown operator.

2. **`pushJoin(...)` / `pushSemiJoin(...)`**

   - If the user’s query has a join that can be co-located (pushed down to the same shard), `LogicalView` merges the join into its subplan by calling `pushDownOpt.pushJoin(...)`.
   - Merges the left’s `plainRows` with the right side’s row references to create a new “plain row type” for the join output.

3. **`optimize()`, `optimizePhySql()`, `optimizeOSS()`**

   - Runs a **HepPlanner** with various rewriting rules (like `FilterMergeRule`, `ProjectMergeRule`, `JoinConditionSimplifyRule`, etc.) to produce a simpler subplan.
   - Removes casts, merges filters, rewrites semi-joins to subqueries, etc.
   - The final RelNode is easier (or more direct) to convert to SQL.

4. **`buildNativeSql(...)` & `getNativeSql(...)`**

   - Actually calls a `RelToSqlConverter` to produce an AST (`SqlNode`) for the subplan.
   - It might store that AST in `nativeSqlNodeHintCache` for reuse.

5. **Partition/Shard Pruning**:

   - `buildPartRoutingPlanInfo(...)`

      – Creates a 

     ```
     PartRoutingPlanInfo
     ```

      object that either:

     1. Collects **Comparatives** for legacy sharding OR
     2. Collects **PartitionPruneSteps** for new partition DBs.

   - Then **`updatePartRoutingPlanInfo(...)`** merges that info into local caches (like `comparatives`, `allPartPruneSteps`).

6. **`couldDynamicPruning()`** / **`dynamicPruningContainsPartitionKey()`**

   - Indicate whether the query has `IN` conditions that might be replaced at runtime with many values, so we do “dynamic pruning.”
   - For large `IN` lists, we only partially parse them or do more advanced pushdown.

7. **`getRelShardInfo(...)`**

   - Produces the final shard/partition info for a specific table index, used by the engine to figure out which shards to hit.
   - If it’s a “new partition table,” it returns a relevant `PartitionPruneStep`; if it’s an older rule-based table, it returns the `comparatives`.

8. **`calculateRowType()`, `getRefByColumnName(...)`, etc.**

   - Dealing with the “plain row type” to figure out how columns map back to table indices or original column names.

------

## 3. How It Fits with `LogicalView`

- **`LogicalView`** owns an instance of **`PushDownOpt`**, passing in the relevant schema name, table name(s), etc.
- Whenever the higher-level planner decides that an operator (Filter, Project, Join, etc.) can be pushed down, it calls `LogicalView.push(...)`, which delegates to `pushDownOpt.push(...)`.
- When finalizing, `LogicalView` calls `pushDownOpt.getNativeSql(...)` or `pushDownOpt.optimize()` to produce the final subplan or native SQL string.
- For partition or shard pruning, `LogicalView` calls methods like `pushDownOpt.getRelShardInfo(...)` or `pushDownOpt.rebuildPartRoutingPlanInfo()`.

------

## 4. The Internal “PlainRows” Mechanism

One unusual but important concept here is **`plainRows`**. Because Calcite will rename or reorder fields internally—for example, if you do:

```
sqlCopiar códigoSELECT t1.col1, t2.col1
FROM table t1
JOIN table t2 ON ...
```

Calcite might rename those columns to avoid collisions (`col1` and `col10`, etc.). Meanwhile, if it’s the **same** table, you can have collisions. So **`plainRows`** tracks:

1. **`plainRowType`**: The original row type (with the original column names).
2. **`plainRefIndex`**: A parallel list of `(index, tableName)` pairs. If the `i`-th field in the final plan actually references the 2nd column of `tableName`, we store `(2, tableName)`. If it’s an expression or a derived column, it stores `(-1, "")`.

This is used for **sharding** or **partition** checks. If we need to figure out, “Which field is `my_shard_key` in the final plan?” we can go back to `plainRefIndex` to see if that final plan field references `(2, tableNameX)`. Then we check if that’s the correct shard column.

------

## 5. Usage in a Single-Node or Hybrid Scenario

If you plan to reuse some of this code for a single-node Percona server:

1. **Keeping or dropping partition logic**
   - If you only have one node, you may not need the partition or `comparatives` logic. You can simplify or ignore calls to `buildPartRoutingPlanInfo()`.
   - But if you do plan to do partial pushdown or sub-sharding on a single node (for caching or partial queries), you can keep the partition-based logic.
2. **Subquery rewriting & Plan rewriting**
   - The code for `optimize()`, `optimizePhySql()`, and the usage of a `HepPlanner` with rules like `FilterMergeRule`, `ProjectMergeRule`, etc., is quite handy if you want to do rewriting at the logical plan level.
   - For single-node usage, you might skip certain distributed rules, but the basic rewriting is still valuable.
3. **Generating a parametric / placeholder query**
   - `buildNativeSql(...)` and `RelToSqlConverter` usage can help you generate a final SQL statement with placeholders (`?`) for parameters.
   - This is helpful if you want to do caching or parameter binding at runtime.
4. **Tracking column references**
   - If you do have queries with repeated references to the same table or complicated expressions, you can keep the `plainRows` logic so you always know which original table/column each final field references. This also helps with any potential local caching or rewriting strategies.

------

## 6. Summary

**`PushDownOpt`** is the engine that “pushes down” all the logical operators into a single subplan and manages the partition/shard metadata. In short:

1. It **accumulates** a **RelNode** subplan (filters, projects, aggregates, sorts, joins).
2. It **rewrites** or **optimizes** that subplan using Calcite’s `HepPlanner` and custom rules.
3. It **extracts** partitioning or sharding info (via `comparatives` or `PartitionPruneStep`), enabling the coordinator to know which shards to query.
4. It **converts** the final subplan to native SQL or XPlan, caching results to avoid rebuilding everything repeatedly.
5. It **maintains** a “plain row” mapping so that sharding columns and references remain correct across merges and renames.

Therefore, **`PushDownOpt`** is a critical helper class for the **`LogicalView`** in PolarDB-X’s distributed (or partially distributed) SQL engine. You can reuse the same architectural ideas if you need to:

- Build an internal RelNode tree.
- Optimize it with “pushdown” transformations or “semi-join to subquery” rewrites.
- Cache or compute partitioning conditions.
- Generate final SQL strings or partial queries for your single-node or multi-node environment.