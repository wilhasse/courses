# Preparation

Install Debian 11 build packages

```bash
sudo apt-get install libaio-dev libssl-dev libncurses5-dev libreadline-dev libcurl4-openssl-dev pkg-config build-essential
sudo apt-get install cmake zlib1g-dev libreadline-dev pkg-config flex bison libaio-dev 
```

In /data directory
download mysql source

Percona MySQL 5.7

```bash
git clone https://github.com/percona/percona-server.git -b 5.7 percona-server-57
cd percona-server-57
git submodule init
git submodule update
```

# Build

```bash
# compile
cmake . -DCMAKE_BUILD_TYPE=Debug -DWITH_DEBUG=1 -DBUILD_CONFIG=mysql_release -DFEATURE_SET=community -DWITH_EMBEDDED_SERVER=OFF -DDOWNLOAD_BOOST=1 -DWITH_BOOST=/home/cslog/boost -B build
cd build
make

# copy to /usr/local/mysql
sudo make install
```

Percona MySQL 8.0

New plugins:

```bash
# kerberos ldap (MySQL 8)
apt-get install libkrb5-dev krb5-multidev libsasl2-2 libsasl2-modules libsasl2-modules-db libldap2-dev libsasl2-dev libsasl2-modules-gssapi-mit
```

```bash
git clone https://github.com/percona/percona-server.git
cd percona-server
git checkout tags/Percona-Server-8.0.39-30
git submodule init
git submodule update
```

Compile

```bash
cmake . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_CONFIG=mysql_release -DFEATURE_SET=community -DFORCE_INSOURCE_BUILD=1 -DDOWNLOAD_BOOST=1 -DWITH_BOOST=/home/cslog/boost_1_77 -DWITH_UNIT_TESTS=OFF -DIGNORE_AIO_CHECK=1 -DWITH_LDAP=system -DLDAP_INCLUDE_DIRS=/usr/include -DLDAP_LIBRARIES=/usr/lib/x86_64-linux-gnu/libldap.so -DLBER_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblber.so -DWITH_KERBEROS=none -DWITH_SASL=OFF -DCMAKE_CXX_FLAGS="-std=c++17 -fno-omit-frame-pointer -DUNIV_LINUX" -B build
cd build
make

# copy to /usr/local/mysql
sudo make install
```

Install only mysql client binary

```bash
sudo apt install default-mysql-client
```
