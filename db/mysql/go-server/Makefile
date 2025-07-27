.PHONY: build run clean test docker

# Build the server binary
build:
	go build -o bin/mysql-server main.go

# Build the debug server binary
build-debug:
	go build -o bin/mysql-debug-server cmd/debug-server/main.go

# Run the server directly
run:
	go run main.go

# Run the debug server with detailed tracing
run-trace:
	go run cmd/debug-server/main.go

# Clean build artifacts
clean:
	rm -rf bin/

# Run tests
test:
	go test ./...

# Download dependencies
deps:
	go mod tidy
	go mod download

# Run with verbose logging
run-verbose:
	LOGLEVEL=debug go run main.go

# Build and run
start: build
	./bin/mysql-server

# Docker commands
docker-build:
	docker build -t mysql-server-example .

docker-run:
	docker run -p 3306:3306 mysql-server-example

# Development helpers
dev-setup: deps
	@echo "Development setup complete"

dev-test-connection:
	@echo "Testing connection (requires mysql client):"
	@echo "mysql -h localhost -P 3306 -u root -e 'SHOW DATABASES;'"

help:
	@echo "Available commands:"
	@echo "  build          - Build the server binary"
	@echo "  build-debug    - Build the debug server binary"
	@echo "  run            - Run the server directly with go run"
	@echo "  run-trace      - Run debug server with detailed execution tracing"
	@echo "  run-verbose    - Run with verbose logging"
	@echo "  clean          - Clean build artifacts"
	@echo "  test           - Run tests"
	@echo "  deps           - Download dependencies"
	@echo "  start          - Build and run the binary"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run in Docker container"
	@echo "  dev-setup      - Set up development environment"
	@echo "  help           - Show this help message"