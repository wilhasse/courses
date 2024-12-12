# Diagram

```mermaid
graph TB
    subgraph Client["Client Applications"]
        App[Applications]
        JDBC[JDBC/MySQL Client]
    end

    subgraph CN["Compute Nodes (CN)"]
        Router[Router/Load Balancer]
        CNServer[CN Server<br>Query Processing]
        MPP[MPP Engine]
        CDC_Producer[CDC Producer]
    end

    subgraph GMS["Global Meta Service (GMS)"]
        MetaDB[MetaDB]
        ConfigMgr[Config Manager]
        StorageMgr[Storage Manager]
        ClusterMgr[Cluster Manager]
    end

    subgraph DN["Data Nodes (DN)"]
        DN1[DN Instance 1]
        DN2[DN Instance 2]
        DN3[DN Instance N]
    end

    subgraph Storage["Storage Layer"]
        MySQL1[MySQL Engine 1]
        MySQL2[MySQL Engine 2]
        MySQLN[MySQL Engine N]
    end

    subgraph CDC["Change Data Capture"]
        CDC_Collector[CDC Collector]
        CDC_Storage[CDC Storage]
        CDC_Consumer[CDC Consumer]
    end

    %% Client connections
    App --> JDBC
    JDBC --> Router
    Router --> CNServer

    %% CN internal
    CNServer --> MPP
    CNServer --> CDC_Producer
    
    %% CN to GMS
    CNServer <--> GMS
    
    %% CN to DN
    MPP --> DN1
    MPP --> DN2
    MPP --> DN3
    
    %% DN to Storage
    DN1 --> MySQL1
    DN2 --> MySQL2
    DN3 --> MySQLN
    
    %% CDC Flow
    CDC_Producer --> CDC_Collector
    CDC_Collector --> CDC_Storage
    CDC_Storage --> CDC_Consumer
    
    %% GMS Management
    GMS --> DN1
    GMS --> DN2
    GMS --> DN3
```

# Description

1. Client Layer:
   - Applications connecting through JDBC/MySQL protocol
   - Router/Load balancer for CN access
2. Compute Nodes (CN):
   - Query processing engine
   - MPP (Massively Parallel Processing) engine
   - CDC Producer for change data capture
3. Global Meta Service (GMS):
   - Configuration management
   - Storage management
   - Cluster management
   - Metadata storage
4. Data Nodes (DN):
   - Multiple DN instances
   - Data storage and processing
   - Direct connection to storage engines
5. Storage Layer:
   - MySQL engines as underlying storage
   - Multiple instances for scalability
6. Change Data Capture (CDC):
   - CDC Producer in CN
   - CDC Collector
   - CDC Storage
   - CDC Consumer for downstream systems

Key interactions shown:

- Client to CN communication
- CN to DN data flow
- GMS management of all components
- CDC data flow through the system
- Storage engine connections