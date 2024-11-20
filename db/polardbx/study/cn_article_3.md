# Introduction

An Interpretation of PolarDB-X Source Codes (3): CDC Code Structure  
https://www.alibabacloud.com/blog/an-interpretation-of-polardb-x-source-codes-3-cdc-code-structure_599438?spm=a2c65.11461447.0.0.7bd2d40dQs5FQR

# GalaxyCDC: PolarDB-X Change Data Capture System

## Core Components

### 1. Architecture Overview
- **Daemon**: Monitoring and control
- **Task**: Core processing engine
- **Dumper**: Data persistence and replication service

### 2. Key Modules
```markdown
- polardbx-cdc-canal: Binlog parsing
- polardbx-cdc-common: Core utilities
- polardbx-cdc-daemon: Process management
- polardbx-cdc-task: Transaction processing
- polardbx-cdc-dumper: Binlog file management
- polardbx-cdc-format: Data transformation
- polardbx-cdc-meta: Metadata management
```

## Database Schema
- binlog_system_config: System parameters
- binlog_task_config: Runtime topology
- binlog_node_info: Node states
- binlog_logic_meta_history: Logical DDL history
- binlog_phy_ddl_history: Physical DDL history

## Development Setup

1. **Prerequisites**
   - JDK 1.8+
   - PolarDB-X instance
   - GalaxySQL source code
   - GalaxyCDC source code

2. **Configuration Steps**
   ```bash
   # Compile GalaxySQL
   mvn install -D maven.test.skip=true -D env=release
   
   # Compile GalaxyCDC
   mvn compile -D maven.test.skip=true -D env=dev
   ```

3. **Launch Services**
   - Start Daemon: `DaemonBootStrap`
   - Start Task: `TaskBootStrap taskName=Final`
   - Start Dumper: `DumperBootStrap taskName=Dumper-1`

## Core Processing Flow
1. Physical binlog processing
2. Local sorting
3. Global sorting/merging
4. Transaction merging
5. Data transmission
6. Storage management
7. Logical binlog generation
8. Replication handling