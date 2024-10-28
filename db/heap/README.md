# Heap Storage Engine

From Percona MySQL 5.7
It uses Percona sources but compile only Heap Storage Engne and uses MySQL Embedded (removed in MySQL 8)

# Percona MySQL 5.7

```bash
git clone https://github.com/percona/percona-server.git -b 5.7 percona-server-57
cd percona-server-57
git submodule init
git submodule update
sudo apt-get install libaio-dev libssl-dev libncurses5-dev libreadline-dev libcurl4-openssl-dev pkg-config
mkdir build
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
```

# Heap

MySQL Include

```bash
cd heap
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

