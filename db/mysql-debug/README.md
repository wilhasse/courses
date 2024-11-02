# Introduction

Help files to set up and debug mysql

- initialize a new data directory
- basic config (my.cnf) and grant all
- run mysqld in compiled version in percona-server

# Preparation

Install packages Debian 12

```bash
sudo apt-get install libldap-dev libsasl2-dev libsasl2-modules-gssapi-mit libkrb5-dev cmake build-essential libaio-dev pkg-config bison gdb rapidjson-dev
```

In /data directory
download mysql source

```bash
git clone https://github.com/percona/percona-server.git
git checkout tags/Percona-Server-8.0.37-29
cd percona-server
git submodule init
git submodule update
```

# Build

## Basic Build Command
```bash
cmake . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_CONFIG=mysql_release -DFEATURE_SET=community -DFORCE_INSOURCE_BUILD=1 -DDOWNLOAD_BOOST=1 -DWITH_BOOST=/home/cslog/boost_1_77 -DWITH_UNIT_TESTS=OFF -DIGNORE_AIO_CHECK=1 -DCMAKE_CXX_FLAGS="-DUNIV_LINUX" -B build
cd build
make
```

## Build Type and Configuration
| Option | Value | Description |
|--------|--------|-------------|
| `CMAKE_BUILD_TYPE` | `RelWithDebInfo` | Optimized build with debugging information |
| `BUILD_CONFIG` | `mysql_release` | Configures build settings for a release build |
| `FEATURE_SET` | `community` | Builds the Community Edition of MySQL |

## Build Location Settings
| Option | Value | Description |
|--------|--------|-------------|
| `FORCE_INSOURCE_BUILD` | `1` | Allows building in the source directory |
| `-B` | `build` | Specifies the build directory |

## Boost Library Settings
| Option | Value | Description |
|--------|--------|-------------|
| `DOWNLOAD_BOOST` | `1` | Automatically downloads Boost during build |
| `WITH_BOOST` | `/home/cslog/boost_1_77` | Specifies Boost library location |

## Testing and System Configuration
| Option | Value | Description |
|--------|--------|-------------|
| `WITH_UNIT_TESTS` | `OFF` | Disables compilation of unit tests |
| `IGNORE_AIO_CHECK` | `1` | Skips AIO library presence verification |

## Compiler Settings
| Option | Value | Description |
|--------|--------|-------------|
| `CMAKE_CXX_FLAGS` | `"-DUNIV_LINUX"` | Defines UNIV_LINUX macro for compilation |

# Commands

Reset data dir

```bash
./initialize_mysql.sh
```

Run Inside screen (it will block terminal)

```bash
./run_mysql.sh
```

Add root grants 

```bash
grant_mysql.sh
```
