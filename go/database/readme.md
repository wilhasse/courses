# Book

Link: https://build-your-own.org/database/

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

- How you add an line to a file ?
- What happens during a crash?

1.4 fsync gotchas

- What is fsync ?
