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
gcc -o example example.c -I. -L/usr/local/lib -linnodb -lpthread -lrt -ldl -Wall -Wextra
```

Run

```bash
./example
```
