#!/bin/bash

echo "=== MySQL Crash Investigation ==="
echo

echo "1. Checking MySQL service status:"
sudo systemctl status mysql | head -20

echo
echo "2. Recent MySQL error log entries:"
sudo tail -30 /var/log/mysql/error.log

echo
echo "3. Checking for core dumps:"
ls -la /var/lib/mysql/core* 2>/dev/null || echo "No core dumps found"

echo
echo "4. To restart MySQL:"
echo "sudo systemctl restart mysql"

echo
echo "5. Checking library dependencies of our plugin:"
ldd /usr/lib/mysql/plugin/mysql_chdb_tvf_embedded.so

echo
echo "=== Likely Issues ==="
echo "The crash is probably due to:"
echo "1. libchdb.so has dependencies that conflict with MySQL"
echo "2. The library is too large (722MB) for MySQL to handle"
echo "3. Symbol conflicts between ClickHouse and MySQL"
echo
echo "=== Solution ==="
echo "We need to use a different approach - perhaps a wrapper process"
echo "instead of directly embedding the huge libchdb.so into MySQL."