# chDB Server API Design

## Overview

Instead of loading 722MB libchdb.so for each query, run a persistent server that:
1. Loads chDB once at startup
2. Keeps ClickHouse data in memory
3. Serves queries via a fast API (HTTP/gRPC)
4. Can run on same machine or separate server

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   MySQL Server  │         │   chDB Server    │
│                 │         │                  │
│  ┌───────────┐  │  HTTP/  │ ┌──────────────┐ │
│  │   UDF     │  │  gRPC   │ │ libchdb.so   │ │
│  │           │──┼─────────┼─│ (loaded once)│ │
│  └───────────┘  │         │ └──────────────┘ │
│                 │         │         │        │
└─────────────────┘         │         ▼        │
                            │ ┌──────────────┐ │
                            │ │ ClickHouse   │ │
                            │ │    Data      │ │
                            │ └──────────────┘ │
                            └──────────────────┘
```

## Option 1: Simple HTTP Server (Easiest)

### Server Code (chdb_http_server.cpp)
```cpp
#include <httplib.h>  // cpp-httplib
#include <dlfcn.h>
#include <json.hpp>
using json = nlohmann::json;

class ChDBServer {
private:
    void* chdb_handle;
    query_stable_v2_fn query_func;
    free_result_v2_fn free_func;
    
public:
    bool init() {
        chdb_handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
        // Load functions...
        return true;
    }
    
    std::string execute_query(const std::string& query) {
        // Execute using loaded chDB
        // Return results
    }
};

int main() {
    ChDBServer chdb;
    chdb.init();
    
    httplib::Server svr;
    
    svr.Post("/query", [&](const httplib::Request& req, httplib::Response& res) {
        json request = json::parse(req.body);
        std::string query = request["query"];
        
        std::string result = chdb.execute_query(query);
        
        json response;
        response["result"] = result;
        response["status"] = "ok";
        
        res.set_content(response.dump(), "application/json");
    });
    
    svr.listen("0.0.0.0", 8123);
}
```

### MySQL UDF Client
```cpp
std::string execute_chdb_query(const std::string& query) {
    CURL* curl = curl_easy_init();
    
    json request;
    request["query"] = query;
    
    // POST to http://localhost:8123/query
    // Return result
}
```

## Option 2: gRPC with Protocol Buffers (Faster)

### chdb_service.proto
```protobuf
syntax = "proto3";

service ChDBService {
    rpc ExecuteQuery(QueryRequest) returns (QueryResponse);
    rpc ExecuteQueryStream(QueryRequest) returns (stream QueryResponse);
}

message QueryRequest {
    string query = 1;
    string format = 2;  // JSON, TSV, CSV
}

message QueryResponse {
    oneof result {
        string data = 1;
        string error = 2;
    }
    int64 rows_read = 3;
    double elapsed_time = 4;
}
```

## Option 3: Lightweight TCP Protocol (Fastest)

### Simple Binary Protocol
```
[4 bytes: message length][1 byte: message type][N bytes: payload]

Message Types:
- 0x01: Query Request
- 0x02: Query Response
- 0x03: Error Response
```

## Performance Comparison

| Approach | Startup Time | Query Latency | Throughput |
|----------|--------------|---------------|------------|
| Current (subprocess) | 2-3 seconds | High | ~1 QPS |
| HTTP Server | Once | ~1-5ms | ~1000 QPS |
| gRPC | Once | ~0.5-2ms | ~5000 QPS |
| TCP Binary | Once | ~0.1-1ms | ~10000 QPS |

## Recommended Implementation

### Phase 1: HTTP Server (Quick Win)
- Easy to implement
- JSON request/response
- Good enough for most use cases
- Can test with curl

### Phase 2: Add Connection Pooling
- Keep multiple chDB instances
- Handle concurrent queries
- Add caching layer

### Phase 3: gRPC for Production
- Better performance
- Streaming support
- Type safety
- Load balancing

## Benefits

1. **722MB loaded once** - Not for every query
2. **Sub-millisecond queries** - After initial load
3. **Scalable** - Can run on separate server
4. **Concurrent** - Handle multiple queries
5. **Cacheable** - Can cache frequent queries
6. **Monitorable** - Add metrics, health checks

## Simple HTTP Implementation Steps

1. Install cpp-httplib:
```bash
git clone https://github.com/yhirose/cpp-httplib.git
```

2. Build the server:
```bash
g++ -o chdb_server chdb_http_server.cpp -ldl -lcurl -pthread -std=c++17
```

3. Update MySQL UDF to use HTTP:
```cpp
// Instead of popen()
CURL* curl = curl_easy_init();
curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8123/query");
// Send query, get response
```

## Docker Deployment

```dockerfile
FROM ubuntu:22.04
COPY libchdb.so /usr/local/lib/
COPY chdb_server /usr/local/bin/
COPY clickhouse_data /data/
EXPOSE 8123
CMD ["/usr/local/bin/chdb_server"]
```

This server approach is much better for production use!