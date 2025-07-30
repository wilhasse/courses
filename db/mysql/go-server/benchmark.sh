#!/bin/bash

# MySQL Server Benchmarking Script
# Compares performance across different storage backends

# Set LD_LIBRARY_PATH for LMDB
export LD_LIBRARY_PATH=$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH

# Configuration
ITERATIONS=100
PORT=3312
HOST=127.0.0.1

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "MySQL Server Benchmark"
echo "======================"
echo "Iterations per test: $ITERATIONS"
echo ""

# Function to start server with specified backend
start_server() {
    local backend=$1
    echo -e "${YELLOW}Starting server with $backend backend...${NC}"
    
    # Kill any existing server
    pkill -f "bin/mysql-server" 2>/dev/null
    sleep 2
    
    # Start server
    ./bin/mysql-server --storage $backend --port $PORT > benchmark_${backend}.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to initialize..."
    sleep 5
    
    # Verify server is running
    if ! mysql -h $HOST -P $PORT -u root -e "SELECT 1" &>/dev/null; then
        echo -e "${RED}Failed to start server with $backend backend${NC}"
        cat benchmark_${backend}.log
        return 1
    fi
    
    echo -e "${GREEN}Server started successfully (PID: $SERVER_PID)${NC}"
    return 0
}

# Function to stop server
stop_server() {
    echo "Stopping server..."
    pkill -f "bin/mysql-server" 2>/dev/null
    sleep 2
}

# Function to run benchmark queries
run_benchmark() {
    local backend=$1
    local query=$2
    local description=$3
    
    echo -e "\n${YELLOW}Benchmark: $description${NC}"
    echo "Backend: $backend"
    echo "Query: $query"
    
    # Warm up
    mysql -h $HOST -P $PORT -u root -e "$query" &>/dev/null
    
    # Run benchmark
    start_time=$(date +%s.%N)
    
    for i in $(seq 1 $ITERATIONS); do
        mysql -h $HOST -P $PORT -u root -e "$query" &>/dev/null
    done
    
    end_time=$(date +%s.%N)
    
    # Calculate results
    total_time=$(echo "$end_time - $start_time" | bc)
    avg_time=$(echo "scale=6; $total_time / $ITERATIONS" | bc)
    qps=$(echo "scale=2; $ITERATIONS / $total_time" | bc)
    
    echo -e "${GREEN}Results:${NC}"
    echo "  Total time: ${total_time}s"
    echo "  Average query time: ${avg_time}s"
    echo "  Queries per second: ${qps}"
    
    # Save results
    echo "$backend,$description,$total_time,$avg_time,$qps" >> benchmark_results.csv
}

# Initialize results file
echo "Backend,Test,Total_Time,Avg_Time,QPS" > benchmark_results.csv

# Benchmark queries
QUERIES=(
    "USE testdb; SELECT * FROM users WHERE id = 1|Point query"
    "USE testdb; SELECT * FROM users|Full table scan"
    "USE testdb; SELECT COUNT(*) FROM products|Aggregation"
    "USE testdb; SELECT * FROM products WHERE price > 50|Range query"
    "USE testdb; SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id|Complex JOIN"
)

# Test each backend
for backend in lmdb; do  # Add more backends as needed: mysql chdb hybrid
    echo -e "\n${YELLOW}=== Testing $backend Backend ===${NC}"
    
    if start_server $backend; then
        # Run each benchmark query
        for query_desc in "${QUERIES[@]}"; do
            IFS='|' read -r query description <<< "$query_desc"
            run_benchmark $backend "$query" "$description"
        done
        
        stop_server
    fi
done

echo -e "\n${YELLOW}=== Benchmark Summary ===${NC}"
echo "Results saved to: benchmark_results.csv"
echo ""
echo "Results by Backend and Test:"
column -t -s',' benchmark_results.csv | head -20

echo -e "\n${GREEN}Benchmark complete!${NC}"