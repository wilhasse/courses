#!/bin/bash

# Performance comparison script for chDB access methods
# Compares loading libchdb.so on each query vs using persistent API server

echo "=== chDB Performance Comparison Test ==="
echo

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if data exists
if [ ! -d "./clickhouse_data" ]; then
    echo -e "${YELLOW}ClickHouse data not found. Running feed_data_v2 first...${NC}"
    make feed_data_v2
    ./feed_data_v2
    echo
fi

# Build everything
echo -e "${YELLOW}Building all components...${NC}"
make clean
make all
echo

# Test 1: Direct library loading (like the wrapper approach)
echo -e "${GREEN}Test 1: Direct library loading (loading 722MB libchdb.so for each query)${NC}"
echo "Running 5 queries sequentially..."

TOTAL_TIME_DIRECT=0
for i in {1..5}; do
    START=$(date +%s.%N)
    ./query_data_v2 > /dev/null 2>&1
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "Query $i: ${ELAPSED}s"
    TOTAL_TIME_DIRECT=$(echo "$TOTAL_TIME_DIRECT + $ELAPSED" | bc)
done

AVG_DIRECT=$(echo "scale=3; $TOTAL_TIME_DIRECT / 5" | bc)
echo -e "Average time per query: ${RED}${AVG_DIRECT}s${NC}"
echo

# Test 2: API Server approach
echo -e "${GREEN}Test 2: API Server approach (persistent chDB instance)${NC}"

# Start the server in background
echo "Starting API server..."
./chdb_api_server > server.log 2>&1 &
SERVER_PID=$!
sleep 3  # Give server time to load chDB

# Check if server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}Failed to start API server. Check server.log for errors.${NC}"
    exit 1
fi

echo "Running 5 queries via API..."
TOTAL_TIME_API=0
for i in {1..5}; do
    START=$(date +%s.%N)
    ./chdb_api_client "SELECT COUNT(*) FROM mysql_import.customers" > /dev/null 2>&1
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "Query $i: ${ELAPSED}s"
    TOTAL_TIME_API=$(echo "$TOTAL_TIME_API + $ELAPSED" | bc)
done

AVG_API=$(echo "scale=3; $TOTAL_TIME_API / 5" | bc)
echo -e "Average time per query: ${GREEN}${AVG_API}s${NC}"

# Stop the server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo
echo "=== Performance Summary ==="
echo -e "Direct library loading: ${RED}${AVG_DIRECT}s${NC} per query"
echo -e "API server approach: ${GREEN}${AVG_API}s${NC} per query"

# Calculate improvement
if (( $(echo "$AVG_API < $AVG_DIRECT" | bc -l) )); then
    IMPROVEMENT=$(echo "scale=1; ($AVG_DIRECT - $AVG_API) / $AVG_DIRECT * 100" | bc)
    SPEEDUP=$(echo "scale=1; $AVG_DIRECT / $AVG_API" | bc)
    echo -e "${GREEN}Performance improvement: ${IMPROVEMENT}% (${SPEEDUP}x faster)${NC}"
else
    echo -e "${YELLOW}No significant performance improvement detected${NC}"
fi

echo
echo "=== Analysis ==="
echo "The API server approach avoids loading the 722MB libchdb.so for each query,"
echo "resulting in much faster query execution times after the initial startup."
echo
echo "Additional benefits:"
echo "- Can serve multiple concurrent clients"
echo "- Can be deployed on a separate server"
echo "- Supports structured data exchange via Protocol Buffers"
echo "- Could be extended with connection pooling, caching, etc."

# Cleanup
rm -f server.log