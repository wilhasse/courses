# Diagram

```mermaid
graph TB
    subgraph Client["Client Layer"]
        App[Applications]
        JDBC[JDBC/MySQL Client]
    end

    subgraph PolarSQL["Polar SQL Layer"]
        direction TB
        Parser[SQL Parser<br>polardbx-parser]
        Optimizer[Query Optimizer<br>polardbx-optimizer]
        
        subgraph Execution["Execution Layer"]
            ParallelExec[Parallel Query<br>Execution Engine]
            Executor[Query Executor<br>polardbx-executor]
            ColumnStore[Columnar Cache<br>Storage]
            ResultCache[Query Result<br>Cache]
        end
    end

    subgraph Backend["Database Backend"]
        XProtocol[X Protocol Plugin]
        PerconaSQL[Percona MySQL Server]
        InnoDB[(InnoDB Engine)]
    end

    %% Client connections
    App --> JDBC
    JDBC --> Parser

    %% Query Flow
    Parser --> Optimizer
    Optimizer --> ParallelExec
    ParallelExec --> Executor
    
    %% Cache and Storage Interactions
    Executor <--> ColumnStore
    Executor <--> ResultCache
    
    %% Backend Flow
    Executor --> XProtocol
    XProtocol --> PerconaSQL
    PerconaSQL --> InnoDB
```

# Description

Client Layer:

- Applications: Any MySQL-compatible client applications
- JDBC/MySQL Client: Standard MySQL protocol connections and connection pooling

Polar SQL Layer:

1. Parser (polardbx-parser)
   - SQL parsing to AST
   - MySQL dialect support
   - Syntax validation
2. Optimizer (polardbx-optimizer)
   - Cost-based query optimization
   - Join ordering and transformations
   - Execution plan generation
3. Execution Components:
   - Parallel Execution: Multi-threaded query processing and result aggregation
   - Query Executor: Plan execution and data flow management
   - Columnar Cache: Column-oriented caching for analytical queries
   - Result Cache: Frequently accessed query results caching

Backend:

- X Protocol: Efficient MySQL client-server protocol
- Percona MySQL: Enhanced MySQL server with performance improvements
- InnoDB: ACID-compliant storage engine with row-level locking

Data Flow:

1. Client → Parser → Optimizer → Parallel Execution → Executor
2. Executor interacts with caches and storage as needed
3. Storage operations handled by InnoDB through X Protocol