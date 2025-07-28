#!/bin/bash

# Build script for MySQL server with LMDB integration
# This script automatically sets up the CGO environment and builds the server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}MySQL Server with LMDB - Build Script${NC}"
echo "========================================"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Go installation
if ! command -v go &> /dev/null; then
    echo -e "${RED}Error: Go is not installed or not in PATH${NC}"
    exit 1
fi

GO_VERSION=$(go version | cut -d' ' -f3)
echo -e "${GREEN}✓ Go found: $GO_VERSION${NC}"

# Check required files
if [ ! -f "lmdb-lib/include/lmdb.h" ]; then
    echo -e "${RED}Error: LMDB header file not found at lmdb-lib/include/lmdb.h${NC}"
    exit 1
fi

if [ ! -f "lmdb-lib/lib/liblmdb.a" ] && [ ! -f "lmdb-lib/lib/liblmdb.so" ]; then
    echo -e "${RED}Error: LMDB library not found in lmdb-lib/lib/${NC}"
    exit 1
fi

echo -e "${GREEN}✓ LMDB library files found${NC}"

# Set up CGO environment
echo -e "${YELLOW}Setting up CGO environment...${NC}"

export CGO_CFLAGS="-I${SCRIPT_DIR}/lmdb-lib/include"
export CGO_LDFLAGS="-L${SCRIPT_DIR}/lmdb-lib/lib -llmdb"

# Platform-specific library path
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${SCRIPT_DIR}/lmdb-lib/lib:$DYLD_LIBRARY_PATH"
    echo -e "${GREEN}✓ macOS library path configured${NC}"
else
    export LD_LIBRARY_PATH="${SCRIPT_DIR}/lmdb-lib/lib:$LD_LIBRARY_PATH"
    echo -e "${GREEN}✓ Linux library path configured${NC}"
fi

echo "CGO_CFLAGS: $CGO_CFLAGS"
echo "CGO_LDFLAGS: $CGO_LDFLAGS"

# Download dependencies
echo -e "${YELLOW}Downloading Go dependencies...${NC}"
go mod tidy
go mod download
echo -e "${GREEN}✓ Dependencies downloaded${NC}"

# Create bin directory
mkdir -p bin

# Build main server
echo -e "${YELLOW}Building main server...${NC}"
go build -o bin/mysql-server main.go
echo -e "${GREEN}✓ Main server built: bin/mysql-server${NC}"

# Build debug server
echo -e "${YELLOW}Building debug server...${NC}"
go build -o bin/mysql-debug-server cmd/debug-server/main.go
echo -e "${GREEN}✓ Debug server built: bin/mysql-debug-server${NC}"

# Create data directory
mkdir -p data

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo -e "${BLUE}To run the server:${NC}"
echo "  1. Export environment variables:"
echo "     export CGO_CFLAGS=\"$CGO_CFLAGS\""
echo "     export CGO_LDFLAGS=\"$CGO_LDFLAGS\""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "     export DYLD_LIBRARY_PATH=\"$DYLD_LIBRARY_PATH\""
else
    echo "     export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
fi
echo ""
echo "  2. Run the server:"
echo "     ./bin/mysql-server          # Main server"
echo "     ./bin/mysql-debug-server    # Debug server with tracing"
echo ""
echo "  3. Or use make commands:"
echo "     make run                    # Run with go run"
echo "     make start                  # Run built binary"
echo ""
echo -e "${BLUE}To connect:${NC}"
echo "     mysql -h 127.0.0.1 -P 3306 -u root"
echo ""
echo -e "${YELLOW}Note: The environment variables are only valid for this shell session.${NC}"
echo -e "${YELLOW}For persistent setup, add them to your ~/.bashrc or ~/.zshrc${NC}"