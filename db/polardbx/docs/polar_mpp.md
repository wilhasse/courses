Below is an expanded, high-level summary of how PolarDB-X executes SQL in parallel, with additional detail on key modules, packages, and classes as found in the open-source codebase. This overview spans the **optimizer**, **scheduler**, **execution**, and **data-transport** layers, showing how PolarDB-X orchestrates multi-node and intra-node parallelism.

------

## 1. Overall Idea

1. **Split the Plan**
    A complex SQL statement is first compiled into a logical plan, which is then broken down into **Fragments** (also called sub-plans). Each Fragment may contain operators such as `Scan`, `Project`, `Filter`, `Agg`, `Join`, etc.

   - Key packages and classes (optimizer layer)

     :

     - `com.alibaba.polardbx.optimizer.core.rel.*`
       - E.g., `LogicalView`, `LogicalJoin`, `LogicalAggregation`, etc.
     - `com.alibaba.polardbx.optimizer.core.planner.*`
       - Houses various planner rules to transform the logical plan.
     - `com.alibaba.polardbx.optimizer.context.OptimizerContext`
       - Manages optimizer-level context and metadata.

2. **Distribute Fragments as Tasks**
    Each Fragment becomes a **Stage**, which is then broken into one or more **Tasks** based on the required concurrency level. These Tasks get scheduled to one or more computing nodes (CNs).

   - Key packages and classes (scheduler layer)

     :

     - `com.alibaba.polardbx.executor.mpp.ExecutionScheduler`
       - Orchestrates top-level scheduling across the cluster.
     - `com.alibaba.polardbx.executor.mpp.metadata.Stage`
       - Represents a sub-plan (Fragment) plus scheduling metadata.
     - `com.alibaba.polardbx.executor.mpp.metadata.Task`
       - Represents the logical execution unit deployed on a CN.

3. **Split by Shard / Segment**
    For operators that scan large data sets, each physical shard or segment on a shard is encapsulated into a **Split**. During runtime, Tasks will **pull** these Splits in batches, so that faster Tasks consume more Splits. This approach balances load across DN (storage) nodes.

   - Key packages and classes (split & shard)

     :

     - `com.alibaba.polardbx.executor.mpp.split.*`
       - E.g., `RemoteSplit`, `JdbcSplit`
       - Encapsulates information about the physical location and offset within a shard.

4. **Intra-Node Parallelism**
    Within each Task on a CN, there is a second-level scheduling (a “two-layer” model). A single Task is broken down into **Pipelines**, each of which spawns multiple **Drivers** (the actual “logical threads”).

   - Key packages and classes (execution layer)

     :

     - `com.alibaba.polardbx.executor.mpp.execution.Pipeline`
       - Subdivides a Task’s plan operators.
     - `com.alibaba.polardbx.executor.mpp.execution.Driver`
       - Represents a *logical* thread which executes the operators in a pipeline.

5. **Data Transmission Layer (DTL)**
    For distributed operators (e.g., in a multi-join scenario), Fragments exchange data using a pull-based approach—downstream operators pull data from upstream buffers. This DTL mechanism ensures flow control and prevents memory overflows.

   - Key packages and classes (DTL)

     :

     - `com.alibaba.polardbx.executor.mpp.operator.ExchangeClient`
       - Pull-based client that requests data from upstream.
     - `com.alibaba.polardbx.executor.mpp.operator.ExchangeSink`
       - Buffers and sends data to downstream ExchangeClients.

------

## 2. Key Execution Steps

1. **Query Coordination**

   - The CN that receives the SQL is designated as the **Query Coordinator**.
   - It runs the optimizer to produce the final logical plan and splits it into Fragments.
   - Creates Stages/Tasks and then initiates cluster-wide scheduling.

2. **One-Layer Scheduling (Multi-Node)**

   - The **ExecutionScheduler** chooses target CNs for each Task, taking into account cluster load.
   - **Splits** are distributed in a “zig-zag” or round-robin fashion across Tasks to prevent a single DN from becoming a bottleneck.

3. **Two-Layer Scheduling (Inside a Node)**

   - Once a Task arrives on a CN, the **Local Scheduler** divides it into Pipelines.
   - Each Pipeline spawns multiple Drivers that share data structures and buffers where possible (e.g., a single build hash table for a HashJoin).
   - This design reduces the network overhead (fewer tasks across nodes) and promotes in-memory parallelism.

4. **Time-Slice Execution**

   - PolarDB-X uses an asynchronous, time-slice model. A Driver runs for a specified time slice (e.g., 500ms).
   - If it blocks on I/O or uses up its slice, it yields and re-enters the scheduling queue.
   - This prevents any single query from monopolizing CPU cores and reduces the chance of scheduling deadlocks.

5. **Resource Isolation**

   - **CPU**: Uses CGroup to isolate TP (transactional) and AP (analytical) queries. For instance, AP queries can be pinned to certain CPU shares and throttled if needed.

   - Memory

     :

     - AP memory can be preempted by TP if the system is under pressure.
     - Large operators (e.g., joins, aggregations) may flush to disk or kill themselves if they exceed memory limits.

   - Key packages and classes (resource mgmt)

     :

     - `com.alibaba.polardbx.executor.mpp.metadata.TaskResource`
       - Manages CPU and memory usage for a Task.
     - `com.alibaba.polardbx.executor.mpp.execution.resource`
       - Contains resource-group or pool-based strategies for concurrency control.

------

## 3. Key Packages & Classes (At a Glance)

1. **Optimizer & Planner**
   - **`com.alibaba.polardbx.optimizer.core.rel`**: Logical operator definitions (`LogicalView`, `LogicalJoin`, etc.).
   - **`com.alibaba.polardbx.optimizer.core.planner`**: Planner rules and transformation logic.
   - **`com.alibaba.polardbx.optimizer.config`**: System-wide config data (e.g., partitioning, SQL modes).
2. **Scheduling**
   - **`com.alibaba.polardbx.executor.mpp.ExecutionScheduler`**: Orchestrates cluster-wide scheduling.
   - **`com.alibaba.polardbx.executor.mpp.metadata.Stage`**: Manages concurrency and lifecycle of each Fragment.
   - **`com.alibaba.polardbx.executor.mpp.metadata.Task`**: Encapsulates the sub-plan assigned to a specific CN.
3. **Execution**
   - **`com.alibaba.polardbx.executor.mpp.execution.Pipeline`**: Defines how operators are linked inside a Task.
   - **`com.alibaba.polardbx.executor.mpp.execution.Driver`**: The “logical thread” that processes data.
   - **`com.alibaba.polardbx.executor.operator.\*`**: Contains specific operator implementations (e.g., `HashJoinOperator`, `ProjectOperator`, `SortOperator`).
4. **Data Exchange (DTL)**
   - **`com.alibaba.polardbx.executor.mpp.operator.ExchangeClient`**: Requests/pulls data from upstream.
   - **`com.alibaba.polardbx.executor.mpp.operator.ExchangeSink`**: Buffers/sends data to downstream tasks.
   - **`com.alibaba.polardbx.executor.mpp.execution.buffer.\*`**: Lower-level buffering strategies (e.g., ring buffers, memory channels).
5. **Splits & Shards**
   - **`com.alibaba.polardbx.executor.mpp.split.JdbcSplit`**: Holds shard-level info (JDBC connection, table, segment offsets).
   - **`com.alibaba.polardbx.executor.mpp.split.RemoteSplit`**: Used for remote data sources or cross-node reads.
6. **Resource Management**
   - **`com.alibaba.polardbx.executor.mpp.metadata.TaskResource`**: Tracks CPU, memory, and network usage.
   - **`com.alibaba.polardbx.executor.mpp.execution.resource.ResourceGroupManager`**: Manages resource pools for concurrency.

------

### Core Takeaway

PolarDB-X’s parallel engine is a *built-in* framework that unifies transactional (TP) and analytical (AP) query execution. By slicing a query into **Fragments → Stages → Tasks → Pipelines → Drivers**, PolarDB-X provides:

- **Multi-Node Parallelism (MPP)**: Spreads Tasks across multiple CNs, load-balancing shards.
- **Intra-Node Parallelism**: Pipelines and Drivers exploit multicore resources on each CN.
- **Adaptive Load Sharing**: Dynamically pulls Splits so that faster nodes do more work.
- **Asynchronous Time-Slice Execution**: Prevents deadlocks and ensures resource fairness.
- **Built-In Resource Isolation**: CGroups and memory preemption protect TP from heavy AP workloads.

All of this happens transparently, so a user’s SQL statement simply benefits from improved performance—whether that query is a quick transactional lookup or a multi-table, TB-scale analytical report.