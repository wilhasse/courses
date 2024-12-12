# Architecture

```mermaid

graph TB
    subgraph Client Layer
        Client(Client Applications)
        MySQL(JDBC Driver)
    end
    
    subgraph Frontend Layer
        NET(Network Protocol<br>polardbx-net)
        Server(Server Core<br>polardbx-server)
    end
    
    subgraph Query Processing
        Parser(SQL Parser<br>polardbx-parser)
        Optimizer(Query Optimizer<br>polardbx-optimizer)
        Calcite(Query Planner<br>polardbx-calcite)
        Executor(Query Executor<br>polardbx-executor)
    end
    
    subgraph Storage Layer
        Transaction(Transaction Manager<br>polardbx-transaction)
        Rule(Sharding Rules<br>polardbx-rule)
        GMS(Metadata Service<br>polardbx-gms)
        ORC(Storage Format<br>polardbx-orc)
    end
    
    subgraph Infrastructure
        Common(Common Utilities<br>polardbx-common)
        RPC(RPC Framework<br>polardbx-rpc)
    end
    
    %% Main flow
    Client --> MySQL --> NET --> Server
    Server --> Parser --> Optimizer --> Calcite --> Executor
    Executor --> Transaction
    Executor --> Rule
    Executor --> ORC
    Server --> GMS
    
    %% Infrastructure dependencies
    Common -.-> NET & Server & Parser & Optimizer & Executor
    RPC -.-> NET & GMS & Transaction
```

# Description

Frontend Layer:

- polardbx-net: Handles network protocols and client connections
- polardbx-server: Core server implementation

Query Processing Chain:

- polardbx-parser: SQL parsing
- polardbx-optimizer: Query optimization
- polardbx-calcite: Query planning using Apache Calcite
- polardbx-executor: Query execution

Storage & Data Management:

- polardbx-transaction: Transaction management
- polardbx-rule: Sharding rules and logic
- polardbx-gms: Global metadata service

Common Services:

- polardbx-common: Shared utilities and common code
- polardbx-rpc: RPC framework for distributed communication

Storage Format Support:

- polardbx-orc: ORC format implementation
- polardbx-orc-tools: Tools for working with ORC format