cmake_minimum_required(VERSION 3.10)
project(innodb_parser C CXX)

# Use C++17 or similar
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Some MySQL-specific definitions
add_definitions(
  -DHAVE_CONFIG_H
  -DMYSQL_SERVER
  -D_GNU_SOURCE
  -DPFS_SKIP
  -DHAVE_BOOL
  -DHAVE_PSI_INTERFACE
  -DHAVE_PSI_MUTEX_INTERFACE
  -DUNIV_PFS_MUTEX
  -DUNIV_PFS_THREAD
  -DMUTEX_SYS
)

# Adjust to your local 5.7 paths
set(MYSQL_SOURCE_DIR "/data/percona-server-57")
set(MYSQL_BUILD_DIR  "/data/percona-server-57/build")

# Where to put parser_main.cc
set(PARSER_SOURCES
  parser_main.cc
  tables_dict.cc
  tables_json.cc
  my_innodb_stubs.cc
)

# Add include directories
include_directories(
  ${MYSQL_SOURCE_DIR}/include
  ${MYSQL_BUILD_DIR}/include
  ${MYSQL_SOURCE_DIR}/sql
  ${MYSQL_SOURCE_DIR}/storage/innobase
  ${MYSQL_SOURCE_DIR}/storage/innobase/include
  ${MYSQL_SOURCE_DIR}/libbinlogevents/export
  ${MYSQL_SOURCE_DIR}/dbug
  # If you store some local .h in PROJECT_SOURCE_DIR/include
  ${PROJECT_SOURCE_DIR}
)

# We might link with core server libs from 5.7:
find_library(MYSQLD_LIB mysqld 
    PATHS ${MYSQL_BUILD_DIR}/archive_output_directory 
    NO_DEFAULT_PATH)
find_library(MYSYS_LIB mysys
    PATHS ${MYSQL_BUILD_DIR}/archive_output_directory 
    NO_DEFAULT_PATH)
find_library(MYSYS_SSL_LIB mysys_ssl
    PATHS ${MYSQL_BUILD_DIR}/archive_output_directory
    NO_DEFAULT_PATH)
find_library(DBUG_LIB dbug
    PATHS ${MYSQL_BUILD_DIR}/archive_output_directory 
    NO_DEFAULT_PATH)

# We can define a stub library if you need PFS stubs or so
add_library(pfs_stub STATIC pfs_stub.c) # if you do that approach
# Or if you don't need PFS stubs, skip

# Build an executable "innodb_parser"
add_executable(innodb_parser
  ${PARSER_SOURCES}
)

# Link required libs
target_link_libraries(innodb_parser
  PRIVATE
    # if you used pfs_stub
    pfs_stub

    ${MYSQLD_LIB}        # main server library
    ${MYSYS_LIB}
    ${MYSYS_SSL_LIB}
    ${DBUG_LIB}

    pthread
    dl
    z
    ssl
    crypto
    aio
    crypt
    rt
)

# Some compile options, ignoring warnings
target_compile_options(innodb_parser PRIVATE -fPIC -w)

# You might want:
set_target_properties(innodb_parser PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
)
