# Percona MySQL study area

```mermaid
flowchart TB
    %% Style definitions
    classDef default fill:#f5f5f5,stroke:#333,stroke-width:2px,color:#1a1a1a
    classDef mainComponent fill:#f0f2f5,stroke:#2f4f4f,stroke-width:2px,color:#1a1a1a
    classDef subComponent fill:#ffffff,stroke:#4a4a4a,stroke-width:1px,color:#1a1a1a
    classDef highlight fill:#f2f4f7,stroke:#2f4f4f,stroke-width:2px,color:#1a1a1a

    %% Connection & Protocol Layer
    subgraph A["Connection & Protocol Layer"]
        style A fill:#f2f4f7,stroke:#2f4f4f,stroke-width:2px
        A1["Client Connections"]
        A2["Communication Protocol"]
    end

    %% SQL Layer
    subgraph B["SQL Layer"]
        style B fill:#f0f2f5,stroke:#2f4f4f,stroke-width:2px
        B1["Parser & Lexical Scanner"]
        B2["Optimizer"]
        B3["SQL Executor"]
    end

    %% Handler/SE Interface
    subgraph C["Handler/SE Interface"]
        style C fill:#f2f4f7,stroke:#2f4f4f,stroke-width:2px
        C1["Handler API"]
        C2["Data Dictionary"]
    end

    %% Storage Engines
    subgraph D["Storage Engines"]
        style D fill:#f0f2f5,stroke:#2f4f4f,stroke-width:2px
        D1["InnoDB (Default engine)"]
        D3["Memory/CSV"]
    end

    %% Replication
    subgraph E["Replication Subsystem"]
        style E fill:#f2f4f7,stroke:#2f4f4f,stroke-width:2px
        E1["Master/Replica (Async)"]
        E2["GTID / Binlog"]
    end

    %% Performance & System Tools
    subgraph F["Performance & System Tools"]
        style F fill:#f0f2f5,stroke:#2f4f4f,stroke-width:2px
        F1["Performance Schema"]
        F2["Information Schema"]
        F3["Sys Schema"]
    end

    %% Percona Enhancements
    subgraph G["Percona Enhancements"]
        style G fill:#f2f4f7,stroke:#2f4f4f,stroke-width:2px
        G1["Percona Toolkit Integration"]
        G2["Xtrabackup"]
    end

    %% Connections
    A --> B
    B1 --> B2
    B2 --> B3
    B3 --> C
    C --> D
    B3 --> E
    B3 --> F
    C --> G

    %% Apply styles to all nodes
    class A1,A2,B1,B2,B3,C1,C2,D1,D3,E1,E2,F1,F2,F3,G1,G2 subComponent
```

# Source code

- [Source code study](./source)

# InnoDB

- [Embedded Innodb 1.0.6](./../innodbtest/README.md)
- [InnoDB File Formats Evolution](./docs/inndb_format.md)
- [InnoDB space (parse .ibd without mysql)](./docs/innodb_space.md)
- [Joining tables and Storage Engine](./docs/join_storage.md)