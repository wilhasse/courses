cmake_minimum_required(VERSION 3.10)
project(innodb_parser C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MySQL/Innodb-specific definitions
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
  -DUNIV_NO_PFS
)

set(MYSQL_8_SOURCE_DIR "/data/percona-server")
set(MYSQL_8_BUILD_DIR  "/data/percona-server/build")

set(PARSER_SOURCES
  parser_main.cc
  tables_dict.cc
  tables_json.cc
  my_innodb_stubs.cc
)

# Add *parent* directories so #include "sql/stream_cipher.h" and #include "ut/ut.h" can be found
include_directories(
  "${MYSQL_8_SOURCE_DIR}"                 # top-level => "sql/stream_cipher.h"
  "${MYSQL_8_SOURCE_DIR}/include"         # "my_sys.h", etc.
  "${MYSQL_8_SOURCE_DIR}/storage/innobase"
  "${MYSQL_8_SOURCE_DIR}/storage/innobase/include"
  "${MYSQL_8_BUILD_DIR}/include"          # generated config files
  "${PROJECT_SOURCE_DIR}"                 # local includes
)

# Try to find the static libraries in the build dir
# If they don't exist, you get NOTFOUND.
find_library(INNOBASE_LIB
    innobase
    PATHS
      "${MYSQL_8_BUILD_DIR}/storage/innobase"
      "${MYSQL_8_BUILD_DIR}"
    NO_DEFAULT_PATH
)
find_library(INNODB_ZIP_LIB
    innodb_zipdecompress
    PATHS
      "${MYSQL_8_BUILD_DIR}/storage/innobase"
      "${MYSQL_8_BUILD_DIR}"
    NO_DEFAULT_PATH
)

add_executable(innodb_parser ${PARSER_SOURCES})

target_link_libraries(innodb_parser
  PRIVATE
    ${INNOBASE_LIB}
    ${INNODB_ZIP_LIB}
    ${MYSYS_LIB}
    ${MYSYS_SSL_LIB}
    ${MYSTRINGS_LIB}
    ${DBUG_LIB}
    pthread
    dl
    z
    ssl
    crypto
    aio
    rt
)

# For fewer warnings, position-independent code, and possibly link-time optimization:
target_compile_options(innodb_parser PRIVATE 
  -fPIC
  -w
  -ffunction-sections
  -fdata-sections
)

set(CMAKE_EXE_LINKER_FLAGS
  "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--as-needed"
)

set_target_properties(innodb_parser PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
