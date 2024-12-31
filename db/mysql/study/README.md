# Percona MySQL study area

```mermaid
flowchart TB
    %% =====================
    %% Subgraph: Connection Interface
    %% =====================
    subgraph A[Connection & Protocol Layer]
        A1(Client Connections)
        A2(Communication Protocol)
    end

    %% =====================
    %% Subgraph: SQL Layer
    %% =====================
    subgraph B[SQL Layer]
        B1[Parser & Lexical Scanner]
        B2[Optimizer]
        B3[SQL Executor]
    end

    %% =====================
    %% Subgraph: Storage Engine Interface
    %% =====================
    subgraph C[Handler/SE Interface]
        C1[Handler API]
        C2[Data Dictionary]
    end

    %% =====================
    %% Subgraph: Storage Engines
    %% =====================
    subgraph D[Storage Engines]
        direction TB
        D1["InnoDB(Default engine)"]
        D3[Memory/CSV]
    end

    %% =====================
    %% Subgraph: Replication
    %% =====================
    subgraph E[Replication Subsystem]
        direction TB
        E1["Master/Replica(Async)"]
        E2[GTID / Binlog]
    end

    %% =====================
    %% Subgraph: Performance & System Tools
    %% =====================
    subgraph F[Performance & System Tools]
        direction TB
        F1[Performance Schema]
        F2[Information Schema]
        F3[Sys Schema]
    end

    %% =====================
    %% Subgraph: Percona-Specific Components
    %% =====================
    subgraph G[Percona Enhancements]
        direction TB
        G1[Percona Toolkit Integration]
        G2[Xtrabackup]
    end

    %% =====================
    %% Links: Data Flow
    %% =====================
    A --> B
    B1 --> B2
    B2 --> B3
    B3 --> C
    C --> D
    B3 --> E
    B3 --> F
    C --> G
```

# Source code

- [Source code study](./source)

# InnoDB

- [Embedded Innodb 1.0.6](./../innodbtest/README.md)
- [InnoDB File Formats Evolution](./docs/inndb_format.md)
- [InnoDB space (parse .ibd without mysql)](./docs/innodb_space.md)
- [Joining tables and Storage Engine](./docs/join_storage.md)