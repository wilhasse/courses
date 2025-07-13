cmake_minimum_required(VERSION 3.20)
project(mysql_chdb_plugin VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags to match your chDB build
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mbmi -mbmi2")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find MySQL development files
set(MYSQL_INCLUDE_DIR "/usr/include/mysql")

if(NOT MYSQL_INCLUDE_DIR)
    message(FATAL_ERROR "MySQL development headers not found. Install mysql-server-dev package:\n"
                        "Ubuntu/Debian: sudo apt-get install mysql-server-dev\n"
                        "CentOS/RHEL: sudo yum install mysql-devel")
endif()

message(STATUS "Found MySQL headers: ${MYSQL_INCLUDE_DIR}")

# chDB configuration
set(CHDB_BUILD_DIR "/home/cslog/chdb/buildlib" CACHE PATH "Path to chDB build directory")
set(CHDB_SOURCE_DIR "/home/cslog/chdb" CACHE PATH "Path to chDB source directory")
set(CHDB_BINARY "${CHDB_BUILD_DIR}/programs/clickhouse" CACHE FILEPATH "Path to chDB binary")

# Verify chDB binary exists
if(NOT EXISTS ${CHDB_BINARY})
    message(FATAL_ERROR "chDB binary not found at: ${CHDB_BINARY}\n"
                        "Please build chDB first or update CHDB_BINARY path")
endif()

message(STATUS "Using chDB binary: ${CHDB_BINARY}")

# Include directories
include_directories(
    ${MYSQL_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Plugin source files
set(PLUGIN_SOURCES
    src/simple_chdb_udf.cpp
)

set(TVF_PLUGIN_SOURCES
    src/test_tvf_plugin.cpp
)

# Create the plugin shared library
add_library(mysql_chdb_plugin SHARED ${PLUGIN_SOURCES})

# Create the table-valued function plugin
add_library(test_tvf_plugin SHARED ${TVF_PLUGIN_SOURCES})

# Compile definitions
target_compile_definitions(mysql_chdb_plugin PRIVATE
    CHDB_BINARY_PATH="${CHDB_BINARY}"
    MYSQL_DYNAMIC_PLUGIN
)

target_compile_definitions(test_tvf_plugin PRIVATE
    MYSQL_DYNAMIC_PLUGIN
)

# Link libraries
target_link_libraries(mysql_chdb_plugin 
    PRIVATE
    pthread
    dl
)

target_link_libraries(test_tvf_plugin 
    PRIVATE
    pthread
)

# Plugin properties
set_target_properties(mysql_chdb_plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "mysql_chdb_plugin"
    POSITION_INDEPENDENT_CODE ON
)

set_target_properties(test_tvf_plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "test_tvf_plugin"
    POSITION_INDEPENDENT_CODE ON
)

# Installation
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/lib/mysql/plugin" CACHE PATH "Installation directory" FORCE)
endif()

install(TARGETS mysql_chdb_plugin test_tvf_plugin
    LIBRARY DESTINATION .
)

# Custom targets
add_custom_target(install-plugin
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
    COMMAND mysql -u root -p -e "CREATE FUNCTION IF NOT EXISTS chdb_query RETURNS STRING SONAME 'mysql_chdb_plugin.so'\\;"
    COMMENT "Installing UDF plugin to MySQL"
)

add_custom_target(uninstall-plugin
    COMMAND mysql -u root -p -e "DROP FUNCTION IF EXISTS chdb_query\\;"
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_INSTALL_PREFIX}/mysql_chdb_plugin.so
    COMMENT "Uninstalling UDF plugin from MySQL"
)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  MySQL Headers: ${MYSQL_INCLUDE_DIR}")
message(STATUS "  chDB Binary: ${CHDB_BINARY}")
message(STATUS "  Install Path: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Flags: ${CMAKE_CXX_FLAGS}")