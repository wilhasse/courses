# Makefile for LMDB Go example

# LMDB library paths
LMDB_INCLUDE := $(shell pwd)/lmdb-lib/include
LMDB_LIB := $(shell pwd)/lmdb-lib/lib

# CGO flags
export CGO_CFLAGS := -I$(LMDB_INCLUDE)
export CGO_LDFLAGS := -L$(LMDB_LIB) -llmdb

# Runtime library path
export LD_LIBRARY_PATH := $(LMDB_LIB):$(LD_LIBRARY_PATH)

.PHONY: all setup build run clean benchmark

all: setup build

setup:
	@echo "Setting up LMDB C library..."
	@./setup-lmdb.sh

build:
	@echo "Building Go program..."
	go build -o lmdb-example main.go

run: build
	@echo "Running example..."
	@./lmdb-example

clean:
	@echo "Cleaning..."
	@rm -f lmdb-example
	@rm -rf testdb

distclean: clean
	@echo "Removing LMDB library..."
	@rm -rf lmdb-lib

test: build
	@echo "Running tests..."
	@./lmdb-example

# Alternative targets
go-build: 
	CGO_CFLAGS="-I$(LMDB_INCLUDE)" CGO_LDFLAGS="-L$(LMDB_LIB) -llmdb" go build

go-run:
	CGO_CFLAGS="-I$(LMDB_INCLUDE)" CGO_LDFLAGS="-L$(LMDB_LIB) -llmdb" LD_LIBRARY_PATH="$(LMDB_LIB):$$LD_LIBRARY_PATH" go run main.go

go-test:
	CGO_CFLAGS="-I$(LMDB_INCLUDE)" CGO_LDFLAGS="-L$(LMDB_LIB) -llmdb" go test ./...

# Benchmark targets
benchmark: setup
	@echo "Running LMDB benchmarks..."
	@./run_benchmarks.sh

benchmark-standalone: setup
	@echo "Running standalone benchmark comparison..."
	@./run_benchmarks.sh --standalone

benchmark-gobench: setup
	@echo "Running Go benchmark tests..."
	@./run_benchmarks.sh --gobench

benchmark-comparison: setup
	@echo "Running comprehensive comparison..."
	@./run_benchmarks.sh