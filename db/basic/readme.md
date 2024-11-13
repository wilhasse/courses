# Book

Link: https://build-your-own.org/database/

[Claude Summary of each chapter](Claude_summary)
# Chapter 1

1.1 Updating files in-lace

- Why not use excell as a database?
- Why databases generally are client-server?
- What happens if I am writing to a file and someone else is reading it ? (zig example)
- What is concurrency ?

1.2 Atomic renaming

Renaming a file to an existing one replaces it atomically; deleting the old file is not needed
(and not correct).

- Why it is not correct? Why is not needed?

1.3 Append-only logs

- Is just adding line to a file solve the problem ?
- How to index data? Read the entire file?
- What happens during a crash?

1.4 fsync gotchas

- What is fsync ?

# Chapter 2

![MySQL B+Tree](images/mysql_btree.png)

![LSM Tree](images/lsm_tree.jpg)

Why log-structured merge tree?

It is not a simple log and it is not a B+Tree , it is log data in layers (tree, it is not related to binary tree or B+Tree)!  
What happens if the base file gets larger?

Updates go to small file first and then at certain point it will be merged into the larger file  
Why do you need multiple levels ? Reduce write amplification  
What is write amplification ? Every time the large file is rewritten when the smaller reaches a threshold  
Write amplification X number of levels (query performance)  

What is in each level? Indexing data structure, could be a sorted array

# Chapter 3

What the differences between a B-Tree and a B+Tree?  
B-Tree stores data in any node  
B+Tree only stores data in leaf nodes  
B+Tree keep all the leaf nodes in a double linked list (facilitate searching)  

Split and Merge nodes when necessary  
What happens after a crash? How to prevent a consistent tree?  
- Copy on write to prevent tree ccorruption (partial writes after a crash). What is the problem with this approach? Write amplification.  
- Double write , copy all modifications on an double write buffer first and then changes the tree. There is an variation that stores the original nodes.    

MySQL uses a B+Tree for the primary and secondary indexes  
- Primary index leaf nodes store data  
- Secondary index leaf nodes store the primary key  

# Chapter 4 and 5

Self-balancing tree data structure with sorted data

Data structures
- BNode - Node in a B+Tree. It's a slice that can be dumped directly to disk.
- BTree - The tree itself. Root pointer and callbacks to managing on-disk pages.

Node format:
- Header (4 bytes)
- Pointers (8 bytes) for leaf nodes these are nulls. Internal node points to child nodes
- Offsets (2 bytes) uses to locate key value pairs within the node
- Key-Value pairs data stored in the node

Key operations:
- Insert
- Delete
- Search

Balancing Operations:
- Splitting when node boecomes full (exceeds BTREE_PAGE_SIZE)
- Merging when node becomes too small (less than BTREE_MIN_FILL)

