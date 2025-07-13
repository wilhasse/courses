CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I/home/cslog/chdb/include $(shell mysql_config --cflags)
LIBS = $(shell mysql_config --libs)

# Define targets
TARGETS = feed_data query_data
COMMON_DEPS = common.h chdb_persist.h

all: $(TARGETS)

# Build feed_data executable
feed_data: feed_data.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build query_data executable
query_data: query_data.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build object files
%.o: %.cpp $(COMMON_DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run feed data
run-feed: feed_data
	./feed_data

# Run query data
run-query: query_data
	./query_data

# Run both in sequence
run-all: run-feed run-query

# Clean build artifacts and data
clean:
	rm -f *.o $(TARGETS) mysql_to_chdb main.o

# Clean persisted data
clean-data:
	rm -rf ./clickhouse_data

# Clean everything
clean-all: clean clean-data

.PHONY: all clean clean-data clean-all run-feed run-query run-all