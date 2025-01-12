cmake_minimum_required(VERSION 3.10)
project(heap_project C CXX)

# Set C++17 standard for C++ files
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add MySQL-specific definitions
add_definitions(
    -DHAVE_CONFIG_H
    -DMYSQL_SERVER
    -D_GNU_SOURCE
    -DDBUG_OFF
    -DPFS_SKIP
    -DHAVE_BOOL
)

# Path to MySQL 5.7 source and build
set(MYSQL_SOURCE_DIR "/data/percona-server-57")
set(MYSQL_BUILD_DIR "/data/percona-server-57/build")

# Add include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${MYSQL_SOURCE_DIR}/include
    ${MYSQL_BUILD_DIR}/include
    ${MYSQL_SOURCE_DIR}/sql
    ${MYSQL_SOURCE_DIR}/storage/heap
    ${MYSQL_SOURCE_DIR}/dbug
)

# Heap engine sources
file(GLOB HEAP_SOURCES 
    "${PROJECT_SOURCE_DIR}/hp_*.c"
)
list(FILTER HEAP_SOURCES EXCLUDE REGEX ".*test[0-9]*.c$")

# Create PFS stub library
add_library(pfs_stub STATIC pfs_stub.c)

# Create heap engine library
add_library(heap_engine STATIC ${HEAP_SOURCES})

# Add test executables
add_executable(heap_test1 hp_test1.c)
add_executable(heap_test2 hp_test2.c)

# Find MySQL libraries
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

# Link libraries for heap_test1
target_link_libraries(heap_test1
    heap_engine
    pfs_stub
    ${MYSQLD_LIB}
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

# Link libraries for heap_test2
target_link_libraries(heap_test2
    heap_engine
    pfs_stub
    ${MYSQLD_LIB}
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

# Add compiler flags
target_compile_options(heap_engine PRIVATE -fPIC -w)
target_compile_options(heap_test1 PRIVATE -fPIC -w)
target_compile_options(heap_test2 PRIVATE -fPIC -w)
target_compile_options(pfs_stub PRIVATE -fPIC -w)

# Make sure PFS stub is built as C
set_source_files_properties(pfs_stub.c PROPERTIES LANGUAGE C)

# Set PIE flags for executables
set_target_properties(heap_test1 heap_test2 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
)
