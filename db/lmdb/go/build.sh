#!/bin/bash

# Build the Go program with LMDB C library paths
export CGO_CFLAGS="-I$(pwd)/lmdb-lib/include"
export CGO_LDFLAGS="-L$(pwd)/lmdb-lib/lib -llmdb"

echo "Building Go program with LMDB..."
go build -o lmdb-example main.go

if [ $? -eq 0 ]; then
    echo "Build successful! Run with: ./lmdb-example"
else
    echo "Build failed!"
    exit 1
fi