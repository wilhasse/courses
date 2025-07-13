#!/bin/bash
cd /home/cslog/courses/db/mysql/tests/mysql-to-chdb-example

echo "Building simple versions..."
g++ -std=c++17 -Wall -Wextra -O2 -c feed_data_simple.cpp -o feed_data_simple.o $(mysql_config --cflags)
g++ -std=c++17 -Wall -Wextra -O2 -o feed_data_simple feed_data_simple.o $(mysql_config --libs) -ldl

g++ -std=c++17 -Wall -Wextra -O2 -c query_data_simple.cpp -o query_data_simple.o $(mysql_config --cflags)
g++ -std=c++17 -Wall -Wextra -O2 -o query_data_simple query_data_simple.o $(mysql_config --libs) -ldl

echo "Build complete. Run with:"
echo "  ./feed_data_simple"
echo "  ./query_data_simple"