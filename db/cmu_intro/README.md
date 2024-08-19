# Course

Fall 2023 
Database Systems  
https://15445.courses.cs.cmu.edu/fall2023/  

Playlist
https://www.youtube.com/playlist?list=PLSE8ODhjZXjbj8BMuIrRcacnQh20hmY9g

# Videos

## 01 - Relational Model & Algebra

Slides:  
https://15445.courses.cs.cmu.edu/fall2023/slides/01-relationalmodel.pdf  

Introduction  
Primary Keys / Foriegn Keys  
Relational Algebras: Select, Projection, Union, Intersection, Differenc , Product, Join.  

## 02 - Modern SQL

Slides:   
https://15445.courses.cs.cmu.edu/fall2023/slides/02-modernsql.pdf  

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

Slides:
https://15445.courses.cs.cmu.edu/fall2023/slides/03-storage1.pdf

Storage Hierarchies  
Disk Oriented (Buffer Pool and Database File)  
Denormalized Tuple Data  
Why not use the OS? Why not use mmap (memory mapping) instead of buffer pool?  
Database Pages  
Heap File  
Slotted Page: Slot Array map "slot" to the page

## 04 - Database Storage Part 2

Slides:
https://15445.courses.cs.cmu.edu/fall2023/slides/04-storage2.pdf  

Tuple-Oriented Storeage   
Log-Structured Storage  
Index-Organized Storage  
Data Representation  
External Value Store  

## 05 - Storage Models & Compression

Slides:  
https://15445.courses.cs.cmu.edu/fall2023/slides/05-storage3.pdf

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
