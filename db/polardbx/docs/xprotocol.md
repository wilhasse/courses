# Introduction

XProtocol uses protobuf for communication between CN (Compute Nodes) and DN (Data Nodes)

# Diagram

```mermaid
classDiagram
    class ClientMessages {
        +SQL_STMT_EXECUTE
        +EXEC_PLAN_READ
    }

    class StmtExecute {
        +bytes stmt
        +bytes stmt_digest 
        +bytes hint
        +Any[] args
        +bool compact_metadata
        +uint64 capabilities
    }

    class ExecPlan {
        +Transaction transaction
        +AnyPlan plan
        +bytes plan_digest
        +Scalar[] parameters
        +SessionVariable[] session_variables
        +bool chunk_result
    }

    class AnyPlan {
        +PlanType plan_type
        +GetPlan get_plan
        +TableScanPlan table_scan_plan
        +Project project
        +Filter filter
        +RangeScan range_scan
        +Aggr aggr
    }

    class SQLLayer {
        +Calcite Parser
        +Plan Generator
    }

    class StorageEngine {
        +Execute SQL
        +Execute Plan
    }

    ClientMessages --> SQLLayer: request
    SQLLayer --> StmtExecute: can generate
    SQLLayer --> ExecPlan: can generate
    StmtExecute --> StorageEngine: direct SQL execution
    ExecPlan --> StorageEngine: optimized plan execution
    ExecPlan --> AnyPlan: contains
    
    note for SQLLayer "Uses Calcite to parse SQL\nand generate execution plans"
    note for StorageEngine "Can handle both raw SQL\nand optimized execution plans"
```

# Description

PolarDB-X SQL Layer:

- Uses Calcite for SQL parsing and planning
- Can send either:
  - Direct SQL statements via `StmtExecute`
  - Optimized execution plans via `ExecPlan`

Communication Paths:

- ClientMessages

Shows the two main types of requests:

  - `SQL_STMT_EXECUTE` (type 12)
  - `EXEC_PLAN_READ` (type 100)

Storage Engine:

- Can handle both types of requests:
  - Execute raw SQL from `StmtExecute`
  - Execute optimized plans from `ExecPlan`

Important Distinction:

- These are parallel paths, not a conversion flow
- The SQL layer decides which path to take based on its optimization strategy
- Using Calcite, it can generate optimized execution plans directly rather than always sending SQL

# Execution plan

```sql
SELECT o.order_id, o.order_date, c.customer_name 
FROM orders o 
JOIN customers c ON o.customer_id = c.id 
WHERE o.order_date > '2024-01-01' 
  AND c.region = 'EAST'
```

```mermaid
classDiagram
    class Project1 {
        PlanType PROJECT
        Fields[order_id, order_date, customer_name]
        projectColumns()
    }

    class Filter1 {
        PlanType FILTER
        Operator AND
        applyWhereConditions()
    }

    class Project2 {
        PlanType PROJECT
        combineJoinResults()
    }

    class Filter2 {
        PlanType FILTER
        Operator EQUAL
        applyJoinCondition()
    }

    class TableScan1 {
        PlanType TABLE_SCAN
        Table orders
        scanTable()
    }

    class TableScan2 {
        PlanType TABLE_SCAN
        Table customers
        scanTable()
    }

    Project1 --> Filter1 : input
    Filter1 --> Project2 : input
    Project2 --> Filter2 : input
    Filter2 --> TableScan1 : left_input
    Filter2 --> TableScan2 : right_input

    note for Project1 "Final Projection:
    SELECT order_id, order_date, customer_name"
    
    note for Filter1 "WHERE Conditions:
    order_date > '2024-01-01'
    AND region = 'EAST'"
    
    note for Filter2 "Join Condition:
    orders.customer_id = customers.id"
```
