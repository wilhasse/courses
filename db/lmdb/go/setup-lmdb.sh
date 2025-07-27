#!/bin/bash

# Download and build LMDB C library
echo "Setting up LMDB C library..."

# Create directory for LMDB
mkdir -p lmdb-lib

# Clone LMDB repository
if [ ! -d "lmdb-lib/lmdb" ]; then
    echo "Cloning LMDB repository..."
    git clone https://github.com/LMDB/lmdb.git lmdb-lib/lmdb
else
    echo "LMDB repository already exists, pulling latest changes..."
    cd lmdb-lib/lmdb
    git pull
    cd ../..
fi

# Build LMDB
echo "Building LMDB..."
cd lmdb-lib/lmdb/libraries/liblmdb
make clean
make

# Create symlinks for easier access
cd ../../../..
mkdir -p lmdb-lib/include
mkdir -p lmdb-lib/lib

# Copy headers and library
cp lmdb-lib/lmdb/libraries/liblmdb/lmdb.h lmdb-lib/include/
cp lmdb-lib/lmdb/libraries/liblmdb/liblmdb.a lmdb-lib/lib/
cp lmdb-lib/lmdb/libraries/liblmdb/liblmdb.so lmdb-lib/lib/ 2>/dev/null || true

echo "LMDB C library setup complete!"
echo "Headers: lmdb-lib/include/"
echo "Library: lmdb-lib/lib/"