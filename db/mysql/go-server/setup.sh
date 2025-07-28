#!/bin/bash

# Universal setup script for MySQL server with LMDB
# Works on Linux, macOS, and Windows (via Git Bash/WSL)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}MySQL Server with LMDB - Universal Setup${NC}"
echo "========================================"

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Linux*)     PLATFORM="Linux";;
        Darwin*)    PLATFORM="macOS";;
        CYGWIN*|MINGW*|MSYS*) PLATFORM="Windows";;
        *)          PLATFORM="Unknown";;
    esac
    echo -e "${GREEN}Platform detected: $PLATFORM${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Go
    if ! command -v go &> /dev/null; then
        echo -e "${RED}❌ Go not found${NC}"
        echo "Please install Go from https://golang.org/"
        exit 1
    fi
    echo -e "${GREEN}✅ Go found: $(go version | cut -d' ' -f3)${NC}"
    
    # Check Git
    if ! command -v git &> /dev/null; then
        echo -e "${RED}❌ Git not found${NC}"
        echo "Please install Git from https://git-scm.com/"
        exit 1
    fi
    echo -e "${GREEN}✅ Git found${NC}"
    
    # Check C compiler
    if command -v gcc &> /dev/null; then
        echo -e "${GREEN}✅ C compiler found: GCC${NC}"
    elif command -v clang &> /dev/null; then
        echo -e "${GREEN}✅ C compiler found: Clang${NC}"
    else
        echo -e "${YELLOW}⚠️  No C compiler found${NC}"
        show_compiler_install_instructions
    fi
    
    # Check Make (optional)
    if command -v make &> /dev/null; then
        echo -e "${GREEN}✅ Make found (recommended)${NC}"
    else
        echo -e "${YELLOW}⚠️  Make not found (optional)${NC}"
    fi
}

# Show compiler installation instructions
show_compiler_install_instructions() {
    echo ""
    echo -e "${YELLOW}C compiler installation:${NC}"
    case $PLATFORM in
        "Linux")
            echo "  Ubuntu/Debian: sudo apt-get install build-essential"
            echo "  CentOS/RHEL:   sudo yum groupinstall 'Development Tools'"
            echo "  Fedora:        sudo dnf groupinstall 'Development Tools'"
            ;;
        "macOS")
            echo "  Install Xcode Command Line Tools:"
            echo "  xcode-select --install"
            ;;
        "Windows")
            echo "  Install one of:"
            echo "  - TDM-GCC: https://jmeubank.github.io/tdm-gcc/"
            echo "  - MinGW-w64: https://www.mingw-w64.org/"
            echo "  - Visual Studio Build Tools"
            ;;
    esac
    echo ""
}

# Setup project
setup_project() {
    echo -e "${YELLOW}Setting up project...${NC}"
    
    # Create directories
    mkdir -p bin data
    
    # Download Go dependencies
    echo "📦 Downloading Go dependencies..."
    go mod tidy
    go mod download
    
    echo -e "${GREEN}✅ Project setup complete${NC}"
}

# Install LMDB
install_lmdb() {
    echo -e "${YELLOW}Setting up LMDB...${NC}"
    
    if [ -f "lmdb-lib/include/lmdb.h" ] && [ -f "lmdb-lib/lib/liblmdb.a" ]; then
        echo -e "${GREEN}✅ LMDB already installed${NC}"
        return 0
    fi
    
    case $PLATFORM in
        "Linux"|"macOS")
            if [ -f "scripts/install-lmdb.sh" ]; then
                echo "🔧 Installing LMDB automatically..."
                chmod +x scripts/install-lmdb.sh
                scripts/install-lmdb.sh
            else
                echo -e "${YELLOW}⚠️  LMDB auto-installer not found${NC}"
                show_manual_lmdb_instructions
            fi
            ;;
        "Windows")
            echo -e "${YELLOW}⚠️  Windows detected - LMDB auto-install not available${NC}"
            show_manual_lmdb_instructions
            ;;
        *)
            echo -e "${YELLOW}⚠️  Unknown platform - manual LMDB setup required${NC}"
            show_manual_lmdb_instructions
            ;;
    esac
}

# Show manual LMDB installation instructions
show_manual_lmdb_instructions() {
    echo ""
    echo -e "${BLUE}Manual LMDB setup required:${NC}"
    echo ""
    echo "Required structure:"
    echo "  lmdb-lib/"
    echo "    include/"
    echo "      lmdb.h"
    echo "    lib/"
    echo "      liblmdb.a (static library)"
    echo "      liblmdb.so (shared library, Linux/macOS)"
    echo "      lmdb.dll (Windows)"
    echo ""
    
    case $PLATFORM in
        "Linux")
            echo "Options for Linux:"
            echo "  1. Use our auto-installer: scripts/install-lmdb.sh"
            echo "  2. System packages: sudo apt-get install liblmdb-dev"
            echo "  3. Download from: https://github.com/LMDB/lmdb/releases"
            ;;
        "macOS")
            echo "Options for macOS:"
            echo "  1. Use our auto-installer: scripts/install-lmdb.sh"
            echo "  2. Homebrew: brew install lmdb"
            echo "  3. Download from: https://github.com/LMDB/lmdb/releases"
            ;;
        "Windows")
            echo "Options for Windows:"
            echo "  1. Pre-compiled binaries (recommended)"
            echo "  2. Compile with MinGW/MSYS2"
            echo "  3. Use Docker for development"
            ;;
    esac
    echo ""
}

# Test build
test_build() {
    echo -e "${YELLOW}Testing build...${NC}"
    
    if go build -o bin/mysql-server main.go; then
        echo -e "${GREEN}✅ Build successful!${NC}"
        return 0
    else
        echo -e "${RED}❌ Build failed${NC}"
        return 1
    fi
}

# Show usage instructions
show_usage() {
    echo ""
    echo -e "${BLUE}🎉 Setup complete!${NC}"
    echo ""
    echo -e "${YELLOW}Quick commands:${NC}"
    
    if command -v make &> /dev/null; then
        echo "  make                  # Quick start (setup + build + run)"
        echo "  make setup            # Setup dependencies"
        echo "  make build            # Build binary"
        echo "  make run              # Run with go run"
        echo "  make help             # Show all commands"
    else
        echo "  go run main.go        # Run server"
        echo "  go build main.go      # Build binary"
        echo "  ./mysql-server        # Run built binary"
    fi
    
    echo ""
    echo -e "${YELLOW}Docker (alternative):${NC}"
    echo "  docker-compose up mysql-server    # Production mode"
    echo "  docker-compose up mysql-dev       # Development mode"
    echo ""
    echo -e "${YELLOW}Connect to server:${NC}"
    echo "  mysql -h 127.0.0.1 -P 3306 -u root"
    echo ""
    
    case $PLATFORM in
        "Windows")
            echo -e "${YELLOW}Windows-specific:${NC}"
            echo "  run.bat               # Quick run"
            echo "  setup.bat             # Windows setup"
            echo ""
            ;;
    esac
}

# Show error recovery
show_error_recovery() {
    echo ""
    echo -e "${RED}❌ Setup encountered issues${NC}"
    echo ""
    echo -e "${YELLOW}Recovery options:${NC}"
    echo "  1. Check prerequisites and try again"
    echo "  2. Use Docker for isolated environment:"
    echo "     docker-compose up mysql-dev"
    echo "  3. Manual LMDB installation (see above)"
    echo "  4. Check documentation in docs/ directory"
    echo ""
}

# Main execution
main() {
    detect_platform
    check_prerequisites
    setup_project
    
    install_lmdb
    
    if test_build; then
        show_usage
    else
        show_error_recovery
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help"
        echo "  --lmdb-only    Only install LMDB"
        echo "  --no-build     Skip build test"
        echo ""
        exit 0
        ;;
    --lmdb-only)
        detect_platform
        install_lmdb
        exit 0
        ;;
    --no-build)
        detect_platform
        check_prerequisites
        setup_project
        install_lmdb
        show_usage
        exit 0
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac