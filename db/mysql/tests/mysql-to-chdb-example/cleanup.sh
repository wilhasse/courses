#!/bin/bash
# Clean up unnecessary files

echo "Removing old simple files..."
rm -f feed_data_simple.cpp
rm -f query_data_simple.cpp
rm -f test_chdb_connection.cpp
rm -f compile_test.sh
rm -f build_simple.sh
rm -f chdb_persist.h

echo "Files cleaned up!"
echo "Remaining source files:"
ls -la *.cpp *.h 2>/dev/null