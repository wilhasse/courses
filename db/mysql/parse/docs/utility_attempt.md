# Goal

Uncompress page using innobase/page/zipdecompress.cc  
The code was refactored in MySQL 8 2015:

```bash
Bug#21405300 - CREATE A INNODB ZIP DECOMPRESSION LIBRARY
This refactoring task is to make page_zip_decompress_low() linkable with
external tools.

This RB is mainly about moving code around.
1. When functions are moved, it doesn't convert functions to doxygen
style.
```

I found out that two utilities: innochecksum and ibd2sdi uses this code

Size of utility:

```bash
slog@mysql-8-src:/data/percona-server/build/runtime_output_directory$ ls -lah innochecksum 
-rwxr-xr-x 1 cslog cslog 12M jan  1 20:27 innochecksum
cslog@mysql-8-src:/data/percona-server/build/runtime_output_directory$ ls -lah ibd2sdi 
-rwxr-xr-x 1 cslog cslog 13M jan  1 20:27 ibd2sdi
cslog@mysql-8-src:/data/percona-server/build/runtime_output_directory$ ls -lah mysqld
-rwxr-xr-x 1 cslog cslog 760M jan  1 16:44 mysqld
```

Created a simple example to call page_zip_decompress_low() and stubbed some part of MySQL to only link with minimal code:

Commit:
https://github.com/wilhasse/percona-server/commit/40615d2ee116143687b99b0acb5b44311bd1f2eb  


How to build only this target I created:

```bash
cd build
cmake --build . --target decompress
```

Error

```bash
cslog@innodb:~$ ./demo_decompress table_c.ibd table.ibd
Found 23552 pages (each 8192 bytes) in input.
2025-01-01 19:09:58 139720832206720  InnoDB: page_zip_decompress 1: 18446744073709551614 8192
page_zip_decompress_low() failed on page 0.
Error reading/decompressing page 0.
```

Adding log to check , how compressed page works:

From the error messages:

```
Found 23552 pages (each 8192 bytes) in input.
...
page_zip_decompress_low() failed on page 0.
Error reading/decompressing page 0.
```

Investigating how to identify if a page is compressed:

[Compressed Table in Innochecksum](./compressed_table.md)


