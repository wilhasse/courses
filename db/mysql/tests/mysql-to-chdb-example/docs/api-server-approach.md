# chDB API Server Approach

## Overview

The chDB API Server provides a persistent service that loads the 722MB libchdb.so once and serves multiple queries efficiently. This approach solves the performance problem of loading the large library for each query.

## Architecture

```
┌─────────────┐     Protocol      ┌─────────────────┐
│ MySQL UDF   │     Buffers       │ chDB API Server │
│   Client    │ ◄──────────────► │  (Port 8125)    │
└─────────────┘    (TCP/Protobuf) └─────────────────┘
                                           │
                                           ▼
                                   ┌──────────────┐
                                   │ libchdb.so   │
                                   │   (722MB)    │
                                   └──────────────┘
```

## Benefits

1. **Performance**: Library loaded once, not for each query
2. **Scalability**: Can serve multiple concurrent clients
3. **Flexibility**: Can run on separate server
4. **Efficiency**: Protocol Buffers for fast serialization
5. **Type Safety**: Structured data exchange

## Implementation

### Protocol Buffer Schema (chdb_api.proto)

```protobuf
message QueryRequest {
    string query = 1;
    OutputFormat format = 2;  // CSV, TSV, JSON, etc.
}

message QueryResponse {
    bool success = 1;
    string error_message = 2;
    repeated Row rows = 3;
    double elapsed_seconds = 7;
}
```

### Server Implementation

The server (`chdb_api_server.cpp`):
- Loads libchdb.so once at startup
- Listens on TCP port 8125
- Uses the stable v2 API (query_stable_v2)
- Parses results into structured protobuf messages
- Handles multiple concurrent connections

### Client Implementation

The client (`chdb_api_client.cpp`):
- Connects to server via TCP
- Sends protobuf-encoded queries
- Receives structured responses
- Measures round-trip time

## Usage

### Starting the Server

```bash
# Build the server
make chdb_api_server

# Start the server (runs on port 8125)
./chdb_api_server

# Or specify custom port
./chdb_api_server 8080
```

### Using the Client

```bash
# Build the client
make chdb_api_client

# Execute a query
./chdb_api_client "SELECT COUNT(*) FROM mysql_import.customers"

# Query with specific output format
./chdb_api_client "SELECT * FROM mysql_import.orders LIMIT 5" TSV
```

### MySQL UDF Integration

To integrate with MySQL, create a UDF that acts as a client:

```sql
-- Future implementation
SELECT chdb_api_query('SELECT COUNT(*) FROM customers');
```

## Performance Comparison

Running `test_performance.sh` shows:

- **Direct loading**: ~2-3 seconds per query (loads 722MB each time)
- **API server**: ~5-50ms per query (after initial startup)
- **Improvement**: 50-100x faster for subsequent queries

## Production Considerations

### Security
- Add authentication/authorization
- Use TLS for encrypted connections
- Implement query whitelisting/validation

### Reliability
- Add connection pooling
- Implement retry logic
- Add health checks
- Use systemd for service management

### Monitoring
- Add metrics collection
- Log query performance
- Monitor memory usage
- Track concurrent connections

## Example systemd Service

```ini
[Unit]
Description=chDB API Server
After=network.target

[Service]
Type=simple
User=chdb
WorkingDirectory=/opt/chdb-server
ExecStart=/opt/chdb-server/chdb_api_server
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Future Enhancements

1. **HTTP/REST API**: Add HTTP endpoint for easier integration
2. **Connection Pooling**: Reuse connections for better performance
3. **Query Caching**: Cache frequent query results
4. **Compression**: Compress large responses
5. **Streaming**: Support streaming for large result sets
6. **Load Balancing**: Support multiple server instances

## Troubleshooting

### Server won't start
- Check if port 8125 is available: `netstat -tulpn | grep 8125`
- Verify libchdb.so path is correct
- Check server.log for errors

### Client connection fails
- Ensure server is running: `ps aux | grep chdb_api_server`
- Check firewall settings
- Verify network connectivity

### Performance issues
- Monitor server memory usage
- Check for concurrent query bottlenecks
- Consider increasing system resources