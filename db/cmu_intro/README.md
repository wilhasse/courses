# Course

Fall 2023 
Database Systems  
https://15445.courses.cs.cmu.edu/fall2023/  

Playlist
https://www.youtube.com/playlist?list=PLSE8ODhjZXjbj8BMuIrRcacnQh20hmY9g

Slides List  
https://15445.courses.cs.cmu.edu/fall2023/slides

# Videos

## 01 - Relational Model & Algebra

Introduction  
Primary Keys / Foriegn Keys  
Relational Algebras: Select, Projection, Union, Intersection, Differenc , Product, Join.  

## 02 - Modern SQL

Aggregate Functions
Group By Clause

Window Functions

Performs a "sliding" calculation across a set of tuples that are related
```sql
SELECT ... FUNC-NAME(...) OVER (...)
 FROM tableName
```

Common Table Expressions (CTE)

```sql
WITH cteName AS (
 SELECT 1
)
SELECT * FROM cteName
```

## 03 - Database Storage Part 1

Storage Hierarchies  
Disk Oriented (Buffer Pool and Database File)  
Denormalized Tuple Data  
Why not use the OS?   
Why not use mmap (memory mapping) instead of buffer pool?  
Database Pages  
Heap File  
Slotted Page: Slot Array map "slot" to the page

## 04 - Database Storage Part 2

Tuple-Oriented Storeage   
Log-Structured Storage  
Index-Organized Storage  
Data Representation  
External Value Store  

## 05 - Storage Models & Compression

Database Workloads  
N-ary Storage Model (NSM)  
Decomposition Store Model (DSM)  
PAX Storage Model (PSM)  

Columnar Compression
- Run-length Encoding
- Bit-Packing Encoding
- Bitmap Encoding
- Delta Encoding
- Incremental Encoding
- Dictionary Encoding

## 06 - Database Memory & Disk I/O Management

Buffer Pool Metadata  
Locks vs. Latches  
Buffer Pool Optimizations  
- Multiple Buffer Pools  
- Pre-feching  
- Scan Sharing  
- Buffer Pool Bypass  
Buffer Pool Replacment Policies
- Least Recently Used (LRU)
- Clock Replacement
- LRU-K
Dirty Pages
OS Page Cache (fsync X O_DIRECT)

## 07 - Hash Tables (CMU Intro to Database Systems)

Static Hash Tables  
Hash Functions  
Linear Probe Hashing  
Non-Unique Keys  
Cukoo Hashing
Chained Hashing  
Extendible Hashing  
Linear Hasing

## 08 - B+Tree Indexes

Visualize B+Tree  
https://cmudb.io/btree  

B+Tree  
Nodes  
Leaf Nodes  

Leaf Nodes Values  
- Record IDs  
- Tuple Data  

B+Tree Insert  
Selection Conditions  
B+Tree Append Record ID  
B+Tree Overflow Leaf Nodes  
Index Scan Page Sorting

B+Tree Design
- Node Size
- Merge Threshold
- Variable-Length Keys
- Intra-Node Search

Optimizations  
- Prefix Compression
- Deduplication
- Suffix Truncation
- Pointer Swizzling
- Bulk Insert
- Buffered Updates

## 09 -Index Concurrency Control

Concurrency Control  

Locks vs. Latches  
Latches Mode
- Read Mode
- Write Mode

Latcnes Implementation
- Test-and-Set Spinlock
- Blocking OS Mutex
- Reader-Writer Locks

Hash Table - Pages Latches  
Hash Table - Slot LLatches  
B+Tree Concurrency Control  
B+Tree - Multi Threaded   
B+Tree - Delete / Insert  
Leaf Node Scan  

## 10 -Sorting

Top-N Heap Sort  
External Merge Sort  
Double Buffering  
Clustered B+Tree  
Aggregations  
Partitions  
Rehash  

## 11- Join Algorithms

Query Plan  
Join Operators  
Operator Output  

Join Algorithms  
- Nested Loop Join
- Block Nested Loop Join
- Index Nested Loop Join
- Sort-Merge Join
- Hash Join

Optimizations: Bloom Filters  
Partitioned Hash Join  
Recursive Partitioning  
Optimizations: Hybrid Hash Join  
