cmake_minimum_required(VERSION 3.10)
project(innodb_parser C)

# Add MySQL-specific definitions
add_definitions(
    -DHAVE_CONFIG_H
    -DMYSQL_SERVER
    -D_GNU_SOURCE
    -DDBUG_OFF
    -DPFS_SKIP
    -DUNIV_LINUX
    -DUNIV_MUST_NOT_INLINE
    -DHAVE_BOOL
)

# Create the InnoDB page library
add_library(innodb_page STATIC innodb_page.c)

# Add parser test executable
add_executable(parser_test parser_test.c)

# Link parser_test with innodb_page
target_link_libraries(parser_test innodb_page)

# Add compiler flags
target_compile_options(parser_test PRIVATE -fPIC -w)
target_compile_options(innodb_page PRIVATE -fPIC -w)

# Set PIE flags for executable
set_target_properties(parser_test PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
)
