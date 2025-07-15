CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I/home/cslog/chdb/programs/local $(shell mysql_config --cflags)
LIBS = $(shell mysql_config --libs) -ldl
PROTOBUF_LIBS = -lprotobuf -pthread

# Define targets
TARGETS = feed_data query_data feed_data_v2 query_data_v2 chdb_api_server chdb_api_client chdb_api_server_simple historico_feeder historico_log historico_log_optimized convert_to_mergetree test_performance execute_sql
COMMON_DEPS = common.h
PROTO_DEPS = chdb_api.pb.h chdb_api.pb.cc

all: $(TARGETS)

# Modern API versions (currently have connection issues)
# Build feed_data executable
feed_data: feed_data.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build query_data executable
query_data: query_data.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# V2 API versions (deprecated but stable)
# Build feed_data_v2 executable
feed_data_v2: feed_data_v2.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build query_data_v2 executable
query_data_v2: query_data_v2.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build historico_feeder executable
historico_feeder: historico_feeder.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build historico_log executable
historico_log: historico_log.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build historico_log_optimized executable
historico_log_optimized: historico_log_optimized.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build convert_to_mergetree executable
convert_to_mergetree: convert_to_mergetree.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build test_performance executable
test_performance: test_performance.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build execute_sql executable
execute_sql: execute_sql.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build object files
%.o: %.cpp $(COMMON_DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run modern API versions
run-feed: feed_data
	./feed_data

run-query: query_data
	./query_data

run-all: run-feed run-query

# Run v2 API versions (recommended for now)
run-feed-v2: feed_data_v2
	./feed_data_v2

run-query-v2: query_data_v2
	./query_data_v2

run-all-v2: run-feed-v2 run-query-v2

# Run historico feeder
run-historico: historico_feeder
	./historico_feeder

# Protobuf API Server and Client
chdb_api.pb.cc chdb_api.pb.h: chdb_api.proto
	protoc --cpp_out=. chdb_api.proto

chdb_api_server: chdb_api_server.o chdb_api.pb.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(PROTOBUF_LIBS)

chdb_api_client: chdb_api_client.o chdb_api.pb.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(PROTOBUF_LIBS)

chdb_api_server.o: chdb_api_server.cpp $(PROTO_DEPS) $(COMMON_DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

chdb_api_client.o: chdb_api_client.cpp $(PROTO_DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

chdb_api.pb.o: chdb_api.pb.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Simple API Server (no protobuf required)
chdb_api_server_simple: chdb_api_server_simple.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) -pthread

chdb_api_server_simple.o: chdb_api_server_simple.cpp $(COMMON_DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run API server
run-server: chdb_api_server
	./chdb_api_server

# Run simple API server
run-server-simple: chdb_api_server_simple
	./chdb_api_server_simple

# Test simple client
test_simple_client: test_simple_client.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Test API client
test-api: chdb_api_client
	@echo "Testing API server with sample queries..."
	./chdb_api_client "SELECT COUNT(*) FROM mysql_import.customers"
	./chdb_api_client "SELECT city, COUNT(*) FROM mysql_import.customers GROUP BY city ORDER BY COUNT(*) DESC" TSV

# Test simple server
test-simple: test_simple_client
	./test_simple_client "SELECT COUNT(*) FROM mysql_import.customers"

# Clean build artifacts and data
clean:
	rm -f *.o $(TARGETS) mysql_to_chdb main.o chdb_api.pb.cc chdb_api.pb.h

# Clean persisted data
clean-data:
	rm -rf ./clickhouse_data

# Clean everything
clean-all: clean clean-data

# Build chdb library (if needed)
build-chdb:
	@echo "To build libchdb.so, run the following commands in the chdb directory:"
	@echo "  cd /home/cslog/chdb"
	@echo "  make build"
	@echo "This will create libchdb.so in the chdb directory"

.PHONY: all clean clean-data clean-all run-feed run-query run-all run-feed-v2 run-query-v2 run-all-v2 build-chdb
