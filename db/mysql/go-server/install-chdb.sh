#!/bin/bash

# Script to install chDB for use with the Go MySQL server
# This handles both AVX and non-AVX processors

set -e

echo "Installing chDB library for Go MySQL server..."
echo ""

# Check processor capabilities
if grep -q avx /proc/cpuinfo; then
    echo "AVX support detected. Installing standard chDB..."
    AVX_SUPPORT=1
else
    echo "No AVX support detected. Will build without AVX..."
    AVX_SUPPORT=0
fi

# Check for required tools
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required"
    exit 1
fi

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y python3-dev build-essential cmake

if [ "$AVX_SUPPORT" -eq 1 ]; then
    # Standard installation for AVX-capable processors
    echo "Installing chDB via pip..."
    pip3 install chdb
else
    # Build from source without AVX
    echo "Building chDB without AVX support..."
    
    # Set environment variables to disable AVX
    export CFLAGS="-march=x86-64 -mtune=generic -mno-avx -mno-avx2"
    export CXXFLAGS="-march=x86-64 -mtune=generic -mno-avx -mno-avx2"
    export ENABLE_AVX=0
    export ENABLE_AVX2=0
    export ENABLE_AVX512=0
    export ARCH_NATIVE=0
    
    # Build from source
    pip3 install --no-binary :all: chdb
fi

# The chdb-go library will find the Python chDB installation automatically
echo ""
echo "Installation complete!"
echo ""
echo "To verify chDB is installed:"
echo "  python3 -c 'import chdb; print(chdb.__version__)'"
echo ""
echo "The Go MySQL server can now use chDB for analytical storage."

# Make script executable
chmod +x "$0"