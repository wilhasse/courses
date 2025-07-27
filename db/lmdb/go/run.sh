#!/bin/bash

# Run the program with the correct library path
export LD_LIBRARY_PATH="$(pwd)/lmdb-lib/lib:$LD_LIBRARY_PATH"

./lmdb-example