# Heap Storage Engine

Extracted from Percona MySQL 5.7 

You need Percona sources (header files and MySQL Embedded) (removed in MySQL 8) 

The reason I am using 5.7 is because embedded server was removed in MySQL 8.

The purpose is to study the heap storage engine and apply some modifications to it. 

Starts with:
- hp_test1.c
- hp_test2.c


# Percona MySQL 5.7

```bash
# Install dependencies
sudo apt-get install libaio-dev libssl-dev libncurses5-dev libreadline-dev libcurl4-openssl-dev pkg-config

# Clone Percona sources
git clone https://github.com/percona/percona-server.git -b 5.7 percona-server-57
cd percona-server-57
git submodule init
git submodule update
mkdir build

# Configure build 
cmake . \
    -DCMAKE_BUILD_TYPE=Debug \
    -DWITH_SSL=system \
    -DWITH_ZLIB=bundled \
    -DMYSQL_MAINTAINER_MODE=1 \
    -DENABLE_DTRACE=0 \
    -DWITH_ZSTD=bundled \
    -DDOWNLOAD_BOOST=1 \
    -DWITH_BOOST=boost \
    -DCMAKE_C_FLAGS="-w" \
    -DCMAKE_CXX_FLAGS="-w" \
    -DDISABLE_PSI_COND=1 \
    -DWITH_UNIT_TESTS=ON \
    -DMYSQL_MAINTAINER_MODE=ON \
    -DWITH_EMBEDDED_SERVER=ON \
    -DWITH_EMBEDDED_SHARED_LIBRARY=ON \
    -B build

# Build
cd build
make
```
# Heap

Link to MySQL headers file

```bash
cd heap
mkdir include
cd include
ln -s /data/percona-server-57/include/my_*.h .
ln -s /data/percona-server-57/include/mysql*.h .
ln -s /data/percona-server-57/include/*.h .
ln -s /data/percona-server-57/include/mysql/*.h mysql/
ln -s /data/percona-server-57/storage/heap/*.h .
ln -s /data/percona-server-57/libbinlogevents/export/binary_log_types.h binary_log_types.h
```

Compile

```bash
./rebuild.sh
```

