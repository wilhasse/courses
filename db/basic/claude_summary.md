# Build Your Own Database From Scratch

This is a chapter-by-chapter summary of "Build Your Own Database From Scratch" by James Smith from build-your-own.org. 

## Table of Contents

### Chapter 00 - Introduction
- **Core Database Concepts**
  - Durability (fsync)
  - Data structures (B-tree/LSM-tree)
  - Storage engines
- **Learning Approach**
  - Emphasis on learning through building
  - Understanding fundamentals over memorizing jargon
  - Why databases are needed beyond simple files

### Chapter 01 - From Files to Databases
- **File Operation Challenges**
  - Problems with in-place updates
  - Atomic operations through renaming
- **Safe Updates**
  - Append-only logs for incremental updates
  - Checksums for corruption detection
- **Durability**
  - fsync usage and considerations
  - Directory synchronization requirements

### Chapter 02 - Indexing Data Structures
- **Query Types**
  - Full scans
  - Point queries
  - Range queries
- **Data Structure Options**
  - Limitations of hashtables
  - B-trees for disk-based storage
  - LSM-trees (Log-Structured Merge trees)
- **Trade-offs**
  - Comparison between B-trees and LSM-trees
  - Write amplification considerations
  - Read performance characteristics

### Chapter 03 - B-Tree Principles
- **Core Concepts**
  - B-trees as balanced n-ary trees
  - B+tree variant and advantages
- **Update Strategies**
  - Copy-on-write techniques
  - Crash safety considerations
- **Tree Operations**
  - Node splitting
  - Node merging
  - Maintaining tree invariants

### Chapter 04 & 05 - Code a B+Tree (Parts I & II)
- **Implementation Details**
  - Node design and format
  - Basic operations
  - Tree maintenance
- **Key Operations**
  - Insertion
  - Deletion
  - Splitting nodes
  - Merging nodes

### Chapter 06 - Append-Only KV Store
- **Key Features**
  - Key-value store using B+trees
  - Two-phase updates
- **File Management**
  - File layout
  - Meta page handling
- **Crash Recovery**
  - Recovery mechanisms
  - Durability guarantees

### Chapter 07 - Free List: Reuse Space
- **Space Management**
  - Free list implementation
  - Page allocation/deallocation
- **Safety Considerations**
  - Crash safety
  - Consistency maintenance
- **Memory Reuse**
  - Page recycling strategies
  - Space efficiency

### Chapter 08 - Rows and Columns
- **Data Organization**
  - Encoding relational data as key-value pairs
  - Primary key implementation
- **Schema Management**
  - Table definitions
  - Column types
- **Multi-Table Support**
  - Table prefixing
  - Namespace management

### Chapter 09 - Range Query
- **Implementation**
  - B+tree iterators
  - Scanning interfaces
- **Data Encoding**
  - Order-preserving encoding
  - Type-specific handling

### Chapter 10 - Secondary Index
- **Index Management**
  - Secondary index implementation
  - Index selection strategies
- **Consistency**
  - Maintaining multiple indexes
  - Update synchronization
- **Atomic Operations**
  - Multi-index updates
  - Consistency guarantees

### Chapter 11 - Atomic Transactions
- **Transaction Support**
  - Basic transaction operations
  - Commit and rollback
- **Consistency**
  - Maintaining data consistency
  - Handling failures
- **Recovery**
  - Transaction recovery
  - State management

### Chapter 12 - Concurrent Readers and Writers
- **Concurrency Model**
  - Multi-reader single-writer implementation
  - Version number management
- **Resource Management**
  - Concurrent free list handling
  - Transaction isolation
- **Safety**
  - Thread safety
  - Deadlock prevention

### Chapter 13 & 14 - Query Language (Parser & Execution)
- **Query Language**
  - SQL-like syntax
  - Expression parsing
- **Execution**
  - Statement processing
  - Query optimization
- **Integration**
  - Combining previous components
  - Complete system implementation

## Conclusion

The book provides a comprehensive journey from basic file operations to a complete database system implementation. It emphasizes practical implementation while explaining the theoretical concepts behind each component. The progression builds understanding from fundamental concepts to complex database features including transactions, concurrency, and query processing.

---
*Note: This summary is based on "Build Your Own Database From Scratch" by James Smith. For the complete content and implementation details, please refer to the original book at build-your-own.org*