# Introduction

An Interpretation of PolarDB-X Source Codes (2): CN Startup Process 
https://www.alibabacloud.com/blog/an-interpretation-of-polardb-x-source-codes-2-cn-startup-process_599437

# PolarDB-X CN Node Startup Process

## Initial Setup
1. **CobarServer Creation**
   - Singleton instance via `CobarServer.getInstance()`

2. **Parameter Loading**
   - Path: `TddlLauncher.main() → CobarServer.init()`
   - Sources (in priority order):
     - Environment variables (highest)
     - Java runtime parameters
     - server.properties (lowest)

## Core Initialization Steps

### 1. Metadata and System Components
- Initialize MetaDB connection pool
- Create/upgrade system tables
- Load instance ID information
- Setup configuration managers:
  - MetaDbConfigManager (metadata changes)
  - MetaDbInstConfigManager (system variables)
  - ConnPoolConfigManager (DN connection pools)
  - StorageHaManager (DN node roles)

### 2. Thread Pools
- managerExecutor: Manager port requests
- killExecutor: Kill commands
- serverExecutor: SQL execution

### 3. Network Layer
- **NIOProcessor**
  - Handles network processing
  - Multiple instances (one per CPU core)
  - Reading and writing threads
  
- **NIOAcceptor**
  - Manages connection establishment
  - Single instance for service ports
  - Binds connections to NIOProcessors

### 4. Additional Services
- MPP Server for CN communication
- CDC Service
- System library initialization
- Logical library (TDataSource) setup
  - Topology initialization
  - DN information gathering
  - Routing configuration
  - Table management
  - Transaction handling
  - Plan cache creation

## Final Steps
1. Warmup (function loading)
2. Network port activation
   - Server port (MySQL compatible)
   - Manager port (internal management)
3. Service availability check