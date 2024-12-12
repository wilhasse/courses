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
        Executor[Query Executor<br>polardbx-executor]
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
    Optimizer --> Executor
    Executor --> XProtocol

    %% Backend Flow
    XProtocol --> PerconaSQL
    PerconaSQL --> InnoDB
```

# Description

Client Layer:

- Standard client applications
- JDBC/MySQL protocol support

Polar SQL Layer (Core Modules):

- SQL Parser: Handles SQL parsing
- Query Optimizer: Optimizes queries
- Query Executor: Executes queries directly

Backend:

- X Protocol Plugin: MySQL's protocol for efficient client-server communication
- Percona MySQL Server: Enhanced MySQL server
- Storage Engine: Direct storage layer