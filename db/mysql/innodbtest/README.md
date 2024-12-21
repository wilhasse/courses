# Embedded InnoDB 

The last time Innobase Oy released an Innodb version  
https://github.com/nextgres/oss-embedded-innodb

Compile

```bash
./configure
./make
./make install
sudo ldconfig

ls -la /usr/local/lib
-rw-r--r--  1 root root 15668192 dez 19 23:33 libinnodb.a
-rwxr-xr-x  1 root root      948 dez 19 23:33 libinnodb.la
lrwxrwxrwx  1 root root       18 dez 19 23:33 libinnodb.so -> libinnodb.so.3.0.0
lrwxrwxrwx  1 root root       18 dez 19 23:33 libinnodb.so.3 -> libinnodb.so.3.0.0
-rwxr-xr-x  1 root root  7030904 dez 19 23:33 libinnodb.so.3.0.0
```

# Example

Compile

```bash
# create_db and test_data
make

# standalone example (from oss-haildb)
gcc -o example example.c -I. -L/usr/local/lib -linnodb -lpthread -lrt -ldl -Wall -Wextra
```

Run

```bash
./example
```

Clean up

```bash
# build file
make clean
# database
./remove_ibd.sh
```

Test

```bash
# new database
./create_db.c
cslog@mysql-8-src:~/courses/db/mysql/innodbtest$ ./create_db 
InnoDB: Mutexes and rw_locks use GCC atomic builtins
InnoDB: The first specified data file ./ibdata1 did not exist:
InnoDB: a new database to be created!
241220 21:53:27  InnoDB: Setting file ./ibdata1 size to 32 MB
InnoDB: Database physically writes the file full: wait...
241220 21:53:27  InnoDB: Log file ./ib_logfile0 did not exist: new to be created
InnoDB: Setting log file ./ib_logfile0 size to 16 MB
InnoDB: Database physically writes the file full: wait...
241220 21:53:27  InnoDB: Log file ./ib_logfile1 did not exist: new to be created
InnoDB: Setting log file ./ib_logfile1 size to 16 MB
InnoDB: Database physically writes the file full: wait...
InnoDB: Doublewrite buffer not found: creating new
InnoDB: Doublewrite buffer created
InnoDB: Creating foreign key constraint system tables
InnoDB: Foreign key constraint system tables created
241220 21:53:27 Embedded InnoDB 1.0.6.6750 started; log sequence number 0
241220 21:53:27  InnoDB: Starting shutdown...
241220 21:53:32  InnoDB: Shutdown completed; log sequence number 47259
cslog@mysql-8-src:~/courses/db/mysql/innodbtest$ 

# searh row
cslog@mysql-8-src:~/courses/db/mysql/innodbtest$ ./test_data 
InnoDB: Mutexes and rw_locks use GCC atomic builtins
241220 21:54:24  InnoDB: highest supported file format is Barracuda.
241220 21:54:24 Embedded InnoDB 1.0.6.6750 started; log sequence number 47259
Found row: id=1234
241220 21:54:24  InnoDB: Starting shutdown...
241220 21:54:29  InnoDB: Shutdown completed; log sequence number 47269
cslog@mysql-8-src:~/courses/db/mysql/innodbtest$ 
cslog@mysql-8-src:~/courses/db/mysql/innodbtest$ 
```
