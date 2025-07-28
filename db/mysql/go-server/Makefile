.PHONY: build run clean test docker setup install-lmdb quick-start

# Variables
BINARY_NAME := mysql-server
DEBUG_BINARY_NAME := mysql-debug-server
BIN_DIR := bin
DATA_DIR := data
LMDB_DIR := lmdb-lib
SCRIPTS_DIR := scripts

# CGO environment variables for LMDB
export CGO_CFLAGS := -I$(shell pwd)/$(LMDB_DIR)/include
export CGO_LDFLAGS := -L$(shell pwd)/$(LMDB_DIR)/lib -llmdb

# Platform-specific library path
ifeq ($(shell uname -s),Darwin)
    export DYLD_LIBRARY_PATH := $(shell pwd)/$(LMDB_DIR)/lib:$(DYLD_LIBRARY_PATH)
else
    export LD_LIBRARY_PATH := $(shell pwd)/$(LMDB_DIR)/lib:$(LD_LIBRARY_PATH)
endif

# Default target
.DEFAULT_GOAL := quick-start

# Check if LMDB is installed
check-lmdb:
	@if [ ! -f "$(LMDB_DIR)/include/lmdb.h" ] || [ ! -f "$(LMDB_DIR)/lib/liblmdb.a" ]; then \
		echo "❌ LMDB not found, installing automatically..."; \
		$(MAKE) install-lmdb; \
	else \
		echo "✅ LMDB found"; \
	fi

# Install LMDB automatically
install-lmdb:
	@echo "🔧 Installing LMDB..."
	@$(SCRIPTS_DIR)/install-lmdb.sh

# Install LMDB from system packages
install-lmdb-system:
	@echo "🔧 Installing LMDB from system packages..."
	@$(SCRIPTS_DIR)/install-lmdb.sh --system

# Force reinstall LMDB
reinstall-lmdb:
	@echo "🔄 Force reinstalling LMDB..."
	@$(SCRIPTS_DIR)/install-lmdb.sh --force

# Complete setup (dependencies + LMDB)
setup: deps check-lmdb
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(DATA_DIR)
	@echo "✅ Setup complete!"

# Quick start - one command to get everything running
quick-start: setup build
	@echo ""
	@echo "🚀 Starting MySQL server..."
	@echo "   Connect with: mysql -h 127.0.0.1 -P 3306 -u root"
	@echo ""
	@./$(BIN_DIR)/$(BINARY_NAME)

# Download Go dependencies
deps:
	@echo "📦 Downloading Go dependencies..."
	@go mod tidy
	@go mod download

# Build the server binary
build: check-lmdb
	@echo "🔨 Building server binary..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/$(BINARY_NAME) main.go
	@echo "✅ Built: $(BIN_DIR)/$(BINARY_NAME)"

# Build the debug server binary
build-debug: check-lmdb
	@echo "🔨 Building debug server binary..."
	@mkdir -p $(BIN_DIR)
	@go build -o $(BIN_DIR)/$(DEBUG_BINARY_NAME) cmd/debug-server/main.go
	@echo "✅ Built: $(BIN_DIR)/$(DEBUG_BINARY_NAME)"

# Build both binaries
build-all: build build-debug

# Run the server directly
run: check-lmdb
	@echo "🚀 Running server with go run..."
	@go run main.go

# Run the debug server with detailed tracing
run-trace: check-lmdb
	@echo "🔍 Running debug server with detailed tracing..."
	@go run cmd/debug-server/main.go

# Run with verbose logging
run-verbose: check-lmdb
	@echo "🚀 Running server with verbose logging..."
	@LOGLEVEL=debug go run main.go

# Build and run binary
start: build
	@echo "🚀 Starting built binary..."
	@./$(BIN_DIR)/$(BINARY_NAME)

# Start debug binary
start-debug: build-debug
	@echo "🔍 Starting debug binary..."
	@./$(BIN_DIR)/$(DEBUG_BINARY_NAME)

# Run tests
test: check-lmdb
	@echo "🧪 Running tests..."
	@go test ./...

# Run tests with coverage
test-coverage: check-lmdb
	@echo "🧪 Running tests with coverage..."
	@go test -cover ./...

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BIN_DIR)/
	@echo "✅ Cleaned $(BIN_DIR)/"

# Clean data directory (reset database)
clean-data:
	@echo "🧹 Cleaning database data..."
	@rm -rf $(DATA_DIR)/
	@echo "✅ Database reset - next run will reinitialize"

# Clean everything including data and LMDB
clean-all: clean
	@echo "🧹 Cleaning all artifacts including data..."
	@rm -rf $(DATA_DIR)/
	@rm -rf $(LMDB_DIR)/
	@echo "✅ Cleaned everything"

# Development helpers
dev-setup: setup
	@echo "🛠️  Development environment ready!"
	@echo ""
	@echo "Quick commands:"
	@echo "  make run           - Run with go run"
	@echo "  make run-trace     - Run debug server"
	@echo "  make build         - Build binary"
	@echo "  make start         - Run built binary"

# Test database connection
test-connection:
	@echo "🔌 Testing database connection..."
	@echo "Trying: mysql -h 127.0.0.1 -P 3306 -u root -e 'SHOW DATABASES;'"
	@mysql -h 127.0.0.1 -P 3306 -u root -e 'SHOW DATABASES;' || echo "❌ Connection failed - is server running?"

# Show system information
info:
	@echo "📊 System Information:"
	@echo "Go version: $(shell go version)"
	@echo "LMDB status: $(shell [ -f $(LMDB_DIR)/include/lmdb.h ] && echo '✅ Installed' || echo '❌ Not found')"
	@echo "Binary status: $(shell [ -f $(BIN_DIR)/$(BINARY_NAME) ] && echo '✅ Built' || echo '❌ Not built')"
	@echo "Data directory: $(shell [ -d $(DATA_DIR) ] && echo '✅ Exists' || echo '❌ Missing')"

# Docker commands
docker-build:
	@echo "🐳 Building Docker image..."
	@docker build -t mysql-server-example .

docker-run:
	@echo "🐳 Running Docker container..."
	@docker run -p 3306:3306 mysql-server-example

# Development commands
dev-watch:
	@echo "👀 Starting file watcher (requires entr)..."
	@echo "Install entr: brew install entr (macOS) or apt-get install entr (Ubuntu)"
	@find . -name "*.go" | entr -r make run

# Benchmark
benchmark: check-lmdb
	@echo "⚡ Running benchmarks..."
	@go test -bench=. ./...

help:
	@echo "🚀 MySQL Server with LMDB - Available Commands:"
	@echo ""
	@echo "🏁 Quick Start:"
	@echo "  quick-start        - One command setup, build, and run"
	@echo "  setup              - Install dependencies and LMDB"
	@echo ""
	@echo "🔧 LMDB Management:"
	@echo "  install-lmdb       - Download and compile LMDB locally"
	@echo "  install-lmdb-system- Use system LMDB packages"
	@echo "  reinstall-lmdb     - Force reinstall LMDB"
	@echo "  check-lmdb         - Verify LMDB installation"
	@echo ""
	@echo "🔨 Building:"
	@echo "  build              - Build main server binary"
	@echo "  build-debug        - Build debug server binary"
	@echo "  build-all          - Build both binaries"
	@echo ""
	@echo "🚀 Running:"
	@echo "  run                - Run with go run"
	@echo "  run-trace          - Run debug server with tracing"
	@echo "  run-verbose        - Run with verbose logging"
	@echo "  start              - Run built main binary"
	@echo "  start-debug        - Run built debug binary"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test               - Run tests"
	@echo "  test-coverage      - Run tests with coverage"
	@echo "  test-connection    - Test MySQL connection"
	@echo "  benchmark          - Run benchmarks"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  dev-setup          - Complete development setup"
	@echo "  dev-watch          - Auto-restart on file changes"
	@echo "  deps               - Download Go dependencies"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean              - Clean build artifacts"
	@echo "  clean-data         - Reset database (clean data directory)"
	@echo "  clean-all          - Clean everything including data"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build       - Build Docker image"
	@echo "  docker-run         - Run in Docker container"
	@echo ""
	@echo "ℹ️  Information:"
	@echo "  info               - Show system information"
	@echo "  help               - Show this help message"