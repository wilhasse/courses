cmake_minimum_required(VERSION 3.10)
project(mysql_to_chdb_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MySQL
find_package(PkgConfig REQUIRED)
pkg_check_modules(MYSQL REQUIRED mysqlclient)

# Add executable
add_executable(mysql_to_chdb main.cpp)

# Include directories
target_include_directories(mysql_to_chdb PRIVATE 
    ${MYSQL_INCLUDE_DIRS}
    /home/cslog/chdb/include  # Adjust this path to your chdb include directory
)

# Link libraries
target_link_libraries(mysql_to_chdb 
    ${MYSQL_LIBRARIES}
    /home/cslog/chdb/build/libchdb.so  # Adjust this path to your chdb library
)

# Add MySQL library directories
target_link_directories(mysql_to_chdb PRIVATE ${MYSQL_LIBRARY_DIRS})

# Compile options
target_compile_options(mysql_to_chdb PRIVATE ${MYSQL_CFLAGS_OTHER})