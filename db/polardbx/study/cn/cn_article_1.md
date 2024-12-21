# Introduction

An Interpretation of PolarDB-X Source Codes (1): CN Code Structure  
https://www.alibabacloud.com/blog/an-interpretation-of-polardb-x-source-codes-1-cn-code-structure_599423

# PolarDB-X Code Architecture

## System Overview
- Complex database system with multiple modules and interfaces
- Follows standard SQL database architecture
- Divided into three main layers: protocol, optimizer, and executor

## Protocol Layer
Located in `polardbx-net` and `polardbx-server` modules:
1. Connection Management (`NIOAcceptor#accept`)
2. Package Parsing (`AbstractConnection#read`)
3. Protocol Parsing (`FrontendCommandHandler#handle`)

## Optimizer
Uses Apache Calcite RBO/CBO framework:

### Key Steps and Interfaces
1. Syntax Parsing: `FastsqlParser#parse`
2. Verification: `SqlConverter#validate`
3. Logical Plan Generation: `SqlConverter#toRel`
4. Logical Plan Optimization: `Planner#optimizeBySqlWriter`
5. Physical Plan Optimization: `Planner#optimizeByPlanEnumerator`

## Executor
### Main Components
1. Execution Entry: `PlanExecutor#execute`
2. Mode Selection: `ExecutorHelper#execute`
3. Execution Modes:
   - Cursor: `AbstractGroupExecutor#executeInner`
   - Local: `LocalExecutionPlanner#plan`
   - MPP: `PlanFragmenter.Fragmenter#buildRootFragment`

### DN Communication
- Cursor mode: `MyJdbcHandler`
- Local/MPP mode: `TableScanClient`

## Code Review Recommendations
1. Start with high-level overview
2. Focus on input/output of each layer
3. Debug simple queries first (e.g., `select 1`)
4. Progress to complex scenarios:
   - Different statement types
   - Protocol variations
   - Large package handling
   - SSL implementation

# SQL Query Plans

## Logical Plan
- Represents WHAT operations need to be done
- Uses abstract operations like "join", "filter", "aggregate"
- Database-engine independent

### Example Query
```sql
SELECT customers.name, SUM(orders.amount) 
FROM customers JOIN orders ON customers.id = orders.customer_id
GROUP BY customers.name
```

### Logical Plan Output
```plain
Aggregate(group=[name], sum=[amount])
  Join(condition=[id = customer_id])
    Scan(table=customers)
    Scan(table=orders)
```

## Physical Plan
- Represents HOW operations will be executed
- Specifies actual algorithms and access methods
- Database-engine specific

### Physical Plan Output
```plain
HashAggregate(group=[name], sum=[amount])
  HashJoin(condition=[id = customer_id])
    TableScan(table=customers, using=btree_index)
    TableScan(table=orders, using=heap_scan) 
```

## Key Differences
1. Physical plan specifies exact join method (HashJoin vs just Join)
2. Physical plan includes access paths (btree_index, heap_scan)
3. Physical plan considers available indexes, memory, data distribution
4. Multiple physical plans can implement one logical plan
