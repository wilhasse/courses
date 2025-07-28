#!/bin/bash

# LMDB Auto-Installation Script
# Downloads, compiles, and installs LMDB locally for the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LMDB_VERSION="0.9.31"
LMDB_URL="https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_${LMDB_VERSION}.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LMDB_DIR="$PROJECT_DIR/lmdb-lib"
TEMP_DIR="/tmp/lmdb-build-$$"

echo -e "${BLUE}LMDB Auto-Installation Script${NC}"
echo "============================="
echo "Version: $LMDB_VERSION"
echo "Target: $LMDB_DIR"
echo ""

# Check if LMDB is already installed
check_lmdb_installed() {
    if [ -f "$LMDB_DIR/include/lmdb.h" ] && [ -f "$LMDB_DIR/lib/liblmdb.a" ]; then
        echo -e "${GREEN}✓ LMDB already installed in $LMDB_DIR${NC}"
        echo -e "${YELLOW}Use --force to reinstall${NC}"
        return 0
    fi
    return 1
}

# Check system dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking system dependencies...${NC}"
    
    # Check for required tools
    local missing_tools=()
    
    if ! command -v make &> /dev/null; then
        missing_tools+=("make")
    fi
    
    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        missing_tools+=("gcc or clang")
    fi
    
    if ! command -v tar &> /dev/null; then
        missing_tools+=("tar")
    fi
    
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        missing_tools+=("curl or wget")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}Error: Missing required tools: ${missing_tools[*]}${NC}"
        echo ""
        echo "On Ubuntu/Debian:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install build-essential curl"
        echo ""
        echo "On CentOS/RHEL/Fedora:"
        echo "  sudo yum groupinstall 'Development Tools'"
        echo "  sudo yum install curl"
        echo ""
        echo "On macOS:"
        echo "  xcode-select --install"
        echo "  brew install curl"
        echo ""
        exit 1
    fi
    
    echo -e "${GREEN}✓ All required tools found${NC}"
}

# Download LMDB source
download_lmdb() {
    echo -e "${YELLOW}Downloading LMDB $LMDB_VERSION...${NC}"
    
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Try multiple URL formats
    local urls=(
        "https://github.com/LMDB/lmdb/archive/refs/tags/LMDB_${LMDB_VERSION}.tar.gz"
        "https://github.com/LMDB/lmdb/archive/LMDB_${LMDB_VERSION}.tar.gz"
        "https://github.com/LMDB/lmdb/releases/download/LMDB_${LMDB_VERSION}/lmdb-LMDB_${LMDB_VERSION}.tar.gz"
    )
    
    local success=false
    for url in "${urls[@]}"; do
        echo "Trying: $url"
        
        if command -v curl &> /dev/null; then
            if curl -L "$url" -o "lmdb.tar.gz" && [ -s "lmdb.tar.gz" ]; then
                # Check if it's actually a gzip file
                if file "lmdb.tar.gz" | grep -q "gzip"; then
                    success=true
                    break
                else
                    echo "Downloaded file is not gzip, trying next URL..."
                    rm -f "lmdb.tar.gz"
                fi
            fi
        elif command -v wget &> /dev/null; then
            if wget "$url" -O "lmdb.tar.gz" && [ -s "lmdb.tar.gz" ]; then
                # Check if it's actually a gzip file
                if file "lmdb.tar.gz" | grep -q "gzip"; then
                    success=true
                    break
                else
                    echo "Downloaded file is not gzip, trying next URL..."
                    rm -f "lmdb.tar.gz"
                fi
            fi
        else
            echo -e "${RED}Error: Neither curl nor wget found${NC}"
            exit 1
        fi
    done
    
    if [ "$success" = false ]; then
        echo -e "${RED}Error: Failed to download LMDB from any URL${NC}"
        echo "Please check your internet connection or try manual installation"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Downloaded LMDB source${NC}"
}

# Extract and prepare source
extract_lmdb() {
    echo -e "${YELLOW}Extracting LMDB source...${NC}"
    
    cd "$TEMP_DIR"
    tar -xzf lmdb.tar.gz
    
    # Find the extracted directory (should be lmdb-LMDB_x.x.x)
    LMDB_SOURCE_DIR=$(find . -maxdepth 1 -type d -name "lmdb-*" | head -n 1)
    
    if [ -z "$LMDB_SOURCE_DIR" ]; then
        echo -e "${RED}Error: Could not find extracted LMDB directory${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Extracted to $LMDB_SOURCE_DIR${NC}"
}

# Compile LMDB
compile_lmdb() {
    echo -e "${YELLOW}Compiling LMDB...${NC}"
    
    cd "$TEMP_DIR/$LMDB_SOURCE_DIR/libraries/liblmdb"
    
    # Clean any previous builds
    make clean 2>/dev/null || true
    
    # Compile with optimizations
    if make -j$(nproc 2>/dev/null || echo 4) CFLAGS="-O2 -fPIC"; then
        echo -e "${GREEN}✓ LMDB compiled successfully${NC}"
    else
        echo -e "${RED}Error: Failed to compile LMDB${NC}"
        exit 1
    fi
}

# Install LMDB locally
install_lmdb() {
    echo -e "${YELLOW}Installing LMDB to project directory...${NC}"
    
    # Create target directories
    mkdir -p "$LMDB_DIR/include"
    mkdir -p "$LMDB_DIR/lib"
    
    cd "$TEMP_DIR/$LMDB_SOURCE_DIR/libraries/liblmdb"
    
    # Copy header file
    cp lmdb.h "$LMDB_DIR/include/"
    
    # Copy library files
    cp liblmdb.a "$LMDB_DIR/lib/"
    
    # Create shared library if not exists
    if [ ! -f liblmdb.so ]; then
        echo -e "${YELLOW}Creating shared library...${NC}"
        gcc -shared -fPIC -o liblmdb.so *.o -ldl
    fi
    cp liblmdb.so "$LMDB_DIR/lib/" 2>/dev/null || echo -e "${YELLOW}Note: Shared library not available${NC}"
    
    # Set appropriate permissions
    chmod 644 "$LMDB_DIR/include/lmdb.h"
    chmod 644 "$LMDB_DIR/lib/"*
    
    echo -e "${GREEN}✓ LMDB installed successfully${NC}"
    echo "  Header: $LMDB_DIR/include/lmdb.h"
    echo "  Static lib: $LMDB_DIR/lib/liblmdb.a"
    [ -f "$LMDB_DIR/lib/liblmdb.so" ] && echo "  Shared lib: $LMDB_DIR/lib/liblmdb.so"
}

# Cleanup temporary files
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    rm -rf "$TEMP_DIR"
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# Verify installation
verify_installation() {
    echo -e "${YELLOW}Verifying installation...${NC}"
    
    if [ ! -f "$LMDB_DIR/include/lmdb.h" ]; then
        echo -e "${RED}Error: Header file not found${NC}"
        return 1
    fi
    
    if [ ! -f "$LMDB_DIR/lib/liblmdb.a" ]; then
        echo -e "${RED}Error: Static library not found${NC}"
        return 1
    fi
    
    # Test header file syntax
    if echo '#include "lmdb.h"' | gcc -I"$LMDB_DIR/include" -x c -c - -o /dev/null 2>/dev/null; then
        echo -e "${GREEN}✓ Header file is valid${NC}"
    else
        echo -e "${RED}Error: Header file appears to be corrupted${NC}"
        return 1
    fi
    
    # Test library linking
    if echo 'int main(){return 0;}' | gcc -L"$LMDB_DIR/lib" -llmdb -x c - -o /dev/null 2>/dev/null; then
        echo -e "${GREEN}✓ Library linking works${NC}"
    else
        echo -e "${YELLOW}Warning: Library linking test failed (may be normal)${NC}"
    fi
    
    echo -e "${GREEN}✓ Installation verified${NC}"
}

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --force     Force reinstallation even if LMDB exists"
    echo "  --system    Try to use system LMDB instead of local install"
    echo "  --help      Show this help message"
    echo ""
    echo "This script downloads, compiles, and installs LMDB locally for the project."
    echo "No system-wide installation or root access required."
}

# Try system LMDB
try_system_lmdb() {
    echo -e "${YELLOW}Checking for system LMDB installation...${NC}"
    
    # Check common locations for LMDB
    local system_include_paths=(
        "/usr/include"
        "/usr/local/include"
        "/opt/homebrew/include"
        "/usr/include/x86_64-linux-gnu"
    )
    
    local system_lib_paths=(
        "/usr/lib"
        "/usr/local/lib"
        "/opt/homebrew/lib"
        "/usr/lib/x86_64-linux-gnu"
        "/usr/lib64"
    )
    
    local found_header=""
    local found_lib=""
    
    # Find header
    for path in "${system_include_paths[@]}"; do
        if [ -f "$path/lmdb.h" ]; then
            found_header="$path/lmdb.h"
            echo -e "${GREEN}✓ Found system LMDB header: $found_header${NC}"
            break
        fi
    done
    
    # Find library
    for path in "${system_lib_paths[@]}"; do
        if [ -f "$path/liblmdb.a" ] || [ -f "$path/liblmdb.so" ]; then
            found_lib="$path"
            echo -e "${GREEN}✓ Found system LMDB library in: $found_lib${NC}"
            break
        fi
    done
    
    if [ -n "$found_header" ] && [ -n "$found_lib" ]; then
        echo -e "${GREEN}System LMDB found! Creating symlinks...${NC}"
        
        mkdir -p "$LMDB_DIR/include"
        mkdir -p "$LMDB_DIR/lib"
        
        ln -sf "$found_header" "$LMDB_DIR/include/lmdb.h"
        
        [ -f "$found_lib/liblmdb.a" ] && ln -sf "$found_lib/liblmdb.a" "$LMDB_DIR/lib/liblmdb.a"
        [ -f "$found_lib/liblmdb.so" ] && ln -sf "$found_lib/liblmdb.so" "$LMDB_DIR/lib/liblmdb.so"
        
        echo -e "${GREEN}✓ System LMDB linked successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}System LMDB not found, will compile from source${NC}"
        return 1
    fi
}

# Main execution
main() {
    local force_install=false
    local use_system=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                force_install=true
                shift
                ;;
            --system)
                use_system=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                usage
                exit 1
                ;;
        esac
    done
    
    # Check if already installed (unless forced)
    if [ "$force_install" = false ] && check_lmdb_installed; then
        exit 0
    fi
    
    # Try system LMDB first if requested
    if [ "$use_system" = true ] && try_system_lmdb; then
        verify_installation
        exit 0
    fi
    
    # Remove existing installation if forcing
    if [ "$force_install" = true ]; then
        echo -e "${YELLOW}Removing existing LMDB installation...${NC}"
        rm -rf "$LMDB_DIR"
    fi
    
    # Install from source
    check_dependencies
    download_lmdb
    extract_lmdb
    compile_lmdb
    install_lmdb
    verify_installation
    cleanup
    
    echo ""
    echo -e "${GREEN}🎉 LMDB installation completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Run: make build"
    echo "  2. Or: go build main.go"
    echo "  3. Start server: ./mysql-server"
    echo ""
    echo -e "${YELLOW}Note: CGO flags are embedded in source code, no environment setup needed!${NC}"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"