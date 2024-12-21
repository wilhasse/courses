# Introduction

An Interpretation of PolarDB-X Source Codes (4): Life of SQL  
https://www.alibabacloud.com/blog/an-interpretation-of-polardb-x-source-codes-4-life-of-sql_599439

# SQL Query Lifecycle in PolarDB-X

## Architecture Overview
- Three main layers: Protocol, Optimizer, and Executor
- Flow: Client Connection → SQL Processing → Result Return

## Protocol Layer
1. **Connection Handling**
   - NIOAcceptor monitors network ports
   - TCP connections bound to NIOProcessor
   - Two threads per processor for read/write

2. **Query Processing**
   - FrontendConnection handles MySQL protocol
   - Command types (e.g., COM_QUERY)
   - Unique traceId generation per query

## Optimizer
### Process Flow
1. **Parser**
   - Lexical analysis
   - Syntax parsing (AST generation)
   - Parameter handling

2. **Plan Management**
   - Plan Cache implementation
   - Execution plan evolution
   - Performance optimization

3. **Validation & Planning**
   - AST to logical plan conversion
   - Semantic checks
   - Type verification

4. **Optimization Steps**
   - SQL Rewriter (RBO)
   - Plan Enumerator (CBO)
   - MPP Planner
   - Post Planner (partition optimization)

## Executor
### Execution Models
1. **Volcano Iterative Model**
2. **Push Model (Pipeline vectorization)**

### Execution Procedures
1. **Cursor**
   - For DML, DDL, DAL statements
   - Volcano model only

2. **Local**
   - For DQL statements
   - Supports both execution models
   - Pipeline splitting
   - Driver generation

3. **MPP**
   - Parallel computing tasks
   - Fragment division
   - Task distribution
   - StageScheduler implementation

### Result Processing
- ResultSetUtil handles result packaging
- Returns data to client via MySQL protocol