CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I/home/cslog/chdb/programs/local $(shell mysql_config --cflags)
LIBS = $(shell mysql_config --libs) -ldl

# Define targets
TARGETS = feed_data query_data feed_data_v2 query_data_v2
COMMON_DEPS = common.h

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

# Clean build artifacts and data
clean:
	rm -f *.o $(TARGETS) mysql_to_chdb main.o

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