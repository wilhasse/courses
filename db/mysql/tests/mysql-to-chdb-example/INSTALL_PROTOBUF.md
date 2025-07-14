# Installing Protocol Buffers for chDB API Server

The chDB API server uses Protocol Buffers for efficient client-server communication. You need to install protobuf to build and run the API server.

## Installation Instructions

### Ubuntu/Debian (Recommended)

```bash
# Update package list
sudo apt-get update

# Install protobuf compiler and development libraries
sudo apt-get install -y protobuf-compiler libprotobuf-dev

# Verify installation
protoc --version
```

### CentOS/RHEL/Fedora

```bash
# Install protobuf
sudo yum install -y protobuf protobuf-devel protobuf-compiler

# Or on newer versions:
sudo dnf install -y protobuf protobuf-devel protobuf-compiler
```

### macOS

```bash
# Using Homebrew
brew install protobuf

# Verify installation
protoc --version
```

### Building from Source (if packages are not available)

```bash
# Download protobuf
wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-cpp-3.21.12.tar.gz
tar -xzf protobuf-cpp-3.21.12.tar.gz
cd protobuf-3.21.12

# Build and install
./configure
make
sudo make install
sudo ldconfig
```

## After Installation

Once protobuf is installed, you can build the API server:

```bash
# Generate C++ files from proto definition
protoc --cpp_out=. chdb_api.proto

# Build the server and client
make chdb_api_server chdb_api_client

# Run the server
./chdb_api_server

# In another terminal, test the client
./chdb_api_client "SELECT COUNT(*) FROM mysql_import.customers"
```

## Troubleshooting

### "protoc: command not found"
- Make sure protobuf-compiler is installed
- Check if /usr/local/bin is in your PATH

### "cannot find -lprotobuf"
- Install libprotobuf-dev (Ubuntu) or protobuf-devel (CentOS)
- Run `sudo ldconfig` to update library cache

### Connection Issues
The client now uses IP address 127.0.0.1 instead of "localhost" to avoid DNS resolution issues. You can also specify custom host and port:

```bash
./chdb_api_client "SELECT 1" CSV 127.0.0.1 8125
```

## Alternative: Using the Server Without Building Client

If you just need the server running and will connect from another application:

1. The server listens on TCP port 8125
2. Protocol: Length-prefixed Protocol Buffers
3. Message format is defined in chdb_api.proto
4. You can connect from any language that supports Protocol Buffers