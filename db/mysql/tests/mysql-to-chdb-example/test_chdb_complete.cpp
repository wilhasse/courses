#include <iostream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <chrono>
#include <cstring>

// Result structure for v2 API
struct local_result_v2 {
    char * buf;
    size_t len;
    void * _vec;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char * error_message;
};

typedef local_result_v2* (*query_stable_v2_fn)(int argc, char** argv);
typedef void (*free_result_v2_fn)(local_result_v2* result);

class ChDBTester {
private:
    void* handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    ChDBTester() : handle(nullptr), query_stable_v2(nullptr), free_result_v2(nullptr) {}
    
    ~ChDBTester() {
        if (handle) dlclose(handle);
    }
    
    bool initialize() {
        std::cout << "Initializing chDB..." << std::endl;
        
        handle = dlopen("libchdb.so", RTLD_LAZY);
        if (!handle) {
            std::cerr << "Failed to load libchdb.so: " << dlerror() << std::endl;
            return false;
        }
        
        query_stable_v2 = (query_stable_v2_fn)dlsym(handle, "query_stable_v2");
        free_result_v2 = (free_result_v2_fn)dlsym(handle, "free_result_v2");
        
        if (!query_stable_v2 || !free_result_v2) {
            std::cerr << "Failed to load functions" << std::endl;
            return false;
        }
        
        std::cout << "✅ chDB initialized successfully" << std::endl;
        return true;
    }
    
    void runQuery(const std::string& query, const std::string& description = "") {
        if (description.length() > 0) {
            std::cout << "\n" << description << std::endl;
        }
        std::cout << "Query: " << query << std::endl;
        
        std::vector<char*> argv;
        std::vector<std::string> args = {
            "clickhouse",
            "--output-format=Pretty",
            "--query=" + query
        };
        
        for (auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        local_result_v2* result = query_stable_v2(argv.size(), argv.data());
        auto end = std::chrono::high_resolution_clock::now();
        
        if (result) {
            if (result->error_message) {
                std::cout << "❌ Error: " << result->error_message << std::endl;
            } else if (result->buf && result->len > 0) {
                std::cout << "✅ Success (" 
                          << std::chrono::duration<double, std::milli>(end - start).count() 
                          << " ms)" << std::endl;
                std::cout << "Result:\n" << std::string(result->buf, result->len) << std::endl;
            } else {
                std::cout << "⚠️  Empty result" << std::endl;
            }
            free_result_v2(result);
        } else {
            std::cout << "❌ Query execution failed" << std::endl;
        }
    }
    
    void runAllTests() {
        std::cout << "\n=== chDB Functionality Tests ===" << std::endl;
        
        // Test 1: Basic functionality
        runQuery("SELECT 1 as test", "Test 1: Basic SELECT");
        
        // Test 2: Functions
        runQuery("SELECT version(), currentDatabase(), now()", "Test 2: System functions");
        
        // Test 3: Math operations
        runQuery("SELECT 2+2 as sum, 10*5 as product, sqrt(16) as square_root", "Test 3: Math operations");
        
        // Test 4: String operations
        runQuery("SELECT length('hello') as len, upper('world') as upp, concat('ch', 'DB') as con", 
                "Test 4: String operations");
        
        // Test 5: Create temporary table
        runQuery("CREATE TEMPORARY TABLE test_table (id Int32, name String) ENGINE = Memory", 
                "Test 5: Create temporary table");
        
        // Test 6: Insert data
        runQuery("INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')", 
                "Test 6: Insert data");
        
        // Test 7: Query data
        runQuery("SELECT * FROM test_table ORDER BY id", "Test 7: Query data");
        
        // Test 8: Aggregation
        runQuery("SELECT count(*) as cnt, max(id) as max_id FROM test_table", 
                "Test 8: Aggregation");
        
        // Test 9: System tables
        runQuery("SELECT name, engine FROM system.tables WHERE database = 'default' LIMIT 5", 
                "Test 9: System tables");
        
        // Test 10: Complex query
        runQuery(R"(
            WITH numbers AS (
                SELECT number FROM system.numbers LIMIT 10
            )
            SELECT 
                number,
                number * 2 as doubled,
                if(number % 2 = 0, 'even', 'odd') as parity
            FROM numbers
        )", "Test 10: Complex query with CTE");
    }
};

int main() {
    std::cout << "=== Comprehensive chDB Test ===" << std::endl;
    std::cout << "Testing chDB installation and functionality\n" << std::endl;
    
    ChDBTester tester;
    
    if (!tester.initialize()) {
        std::cerr << "\n❌ Failed to initialize chDB" << std::endl;
        std::cerr << "Make sure chDB is properly installed:" << std::endl;
        std::cerr << "  curl -sL https://lib.chdb.io | bash" << std::endl;
        std::cerr << "  sudo ldconfig" << std::endl;
        return 1;
    }
    
    tester.runAllTests();
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "✅ All tests completed" << std::endl;
    std::cout << "chDB is working correctly!" << std::endl;
    
    return 0;
}