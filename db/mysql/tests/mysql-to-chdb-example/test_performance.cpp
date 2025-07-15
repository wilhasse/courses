#include <iostream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <chrono>
#include "common.h"

struct local_result_v2 {
    char * buf;
    size_t len;
    void * _vec;
    double elapsed;
    uint64_t rows_read;
    uint64_t bytes_read;
    char * error_message;
};

typedef struct local_result_v2* (*query_stable_v2_fn)(int argc, char** argv);
typedef void (*free_result_v2_fn)(struct local_result_v2* result);

class PerformanceTester {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    PerformanceTester() : chdb_handle(nullptr) {}
    
    ~PerformanceTester() {
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool loadChdbLibrary() {
        chdb_handle = dlopen("libchdb.so", RTLD_LAZY);
        if (!chdb_handle) {
            std::cerr << "Failed to load libchdb.so: " << dlerror() << std::endl;
            return false;
        }
        
        query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
        free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
        
        if (!query_stable_v2 || !free_result_v2) {
            std::cerr << "Failed to load functions: " << dlerror() << std::endl;
            return false;
        }
        
        return true;
    }
    
    struct local_result_v2* executeQuery(const std::string& query) {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        return query_stable_v2(argv.size(), argv.data());
    }
    
    void runTest(const std::string& test_name, const std::string& query_log, const std::string& query_mt) {
        std::cout << "\n" << test_name << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        // Test Log table
        auto start = std::chrono::high_resolution_clock::now();
        auto result = executeQuery(query_log);
        auto end = std::chrono::high_resolution_clock::now();
        auto log_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::string log_output;
        if (result) {
            if (result->buf) {
                log_output = std::string(result->buf);
                // Limit output for display
                if (log_output.length() > 200) {
                    log_output = log_output.substr(0, 200) + "...";
                }
            }
            free_result_v2(result);
        }
        
        // Test MergeTree table
        start = std::chrono::high_resolution_clock::now();
        result = executeQuery(query_mt);
        end = std::chrono::high_resolution_clock::now();
        auto mt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        std::string mt_output;
        if (result) {
            if (result->buf) {
                mt_output = std::string(result->buf);
                if (mt_output.length() > 200) {
                    mt_output = mt_output.substr(0, 200) + "...";
                }
            }
            free_result_v2(result);
        }
        
        std::cout << "Log Engine: " << log_ms << "ms" << std::endl;
        if (!log_output.empty()) std::cout << "  Result: " << log_output << std::endl;
        
        std::cout << "MergeTree: " << mt_ms << "ms" << std::endl;
        if (!mt_output.empty()) std::cout << "  Result: " << mt_output << std::endl;
        
        if (mt_ms > 0) {
            float speedup = (float)log_ms / mt_ms;
            std::cout << "Speedup: " << speedup << "x";
            if (speedup < 1.5) {
                std::cout << " (minimal improvement - query might be too simple)";
            } else if (speedup > 10) {
                std::cout << " (excellent improvement!)";
            }
            std::cout << std::endl;
        }
    }
    
    void runAllTests() {
        std::cout << "\n=== ClickHouse Log vs MergeTree Performance Tests ===" << std::endl;
        
        // First, get table stats
        auto result = executeQuery("SELECT COUNT(*), MIN(id_contr), MAX(id_contr) FROM mysql_import.historico");
        if (result && result->buf) {
            std::cout << "Log table stats: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*), MIN(id_contr), MAX(id_contr) FROM mysql_import.historico_mt");
        if (result && result->buf) {
            std::cout << "MergeTree table stats: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        // Test 1: Simple count with range filter (should show big improvement)
        runTest("Test 1: Count with specific ID range",
                "SELECT COUNT(*) FROM mysql_import.historico WHERE id_contr BETWEEN 5000000 AND 6000000",
                "SELECT COUNT(*) FROM mysql_import.historico_mt WHERE id_contr BETWEEN 5000000 AND 6000000");
        
        // Test 2: Point lookup (should be much faster on MergeTree)
        runTest("Test 2: Point lookup by primary key",
                "SELECT * FROM mysql_import.historico WHERE id_contr = 1234567 AND seq = 1",
                "SELECT * FROM mysql_import.historico_mt WHERE id_contr = 1234567 AND seq = 1");
        
        // Test 3: Aggregation with filter
        runTest("Test 3: Aggregation with filter",
                "SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico WHERE id_contr BETWEEN 1000000 AND 2000000 GROUP BY codigo ORDER BY cnt DESC LIMIT 10",
                "SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico_mt WHERE id_contr BETWEEN 1000000 AND 2000000 GROUP BY codigo ORDER BY cnt DESC LIMIT 10");
        
        // Test 4: Date range query
        runTest("Test 4: Date range query",
                "SELECT COUNT(*), MIN(data), MAX(data) FROM mysql_import.historico WHERE data >= '2024-01-01' AND data < '2024-02-01'",
                "SELECT COUNT(*), MIN(data), MAX(data) FROM mysql_import.historico_mt WHERE data >= '2024-01-01' AND data < '2024-02-01'");
        
        // Test 5: Complex query with multiple conditions
        runTest("Test 5: Complex query with joins and filters",
                "SELECT id_funcionario, COUNT(DISTINCT id_contr) as contracts, COUNT(*) as events FROM mysql_import.historico WHERE id_contr BETWEEN 2000000 AND 3000000 AND codigo IN (51, 188, 132) GROUP BY id_funcionario HAVING events > 100 ORDER BY contracts DESC LIMIT 20",
                "SELECT id_funcionario, COUNT(DISTINCT id_contr) as contracts, COUNT(*) as events FROM mysql_import.historico_mt WHERE id_contr BETWEEN 2000000 AND 3000000 AND codigo IN (51, 188, 132) GROUP BY id_funcionario HAVING events > 100 ORDER BY contracts DESC LIMIT 20");
        
        // Test 6: Full table scan (should show minimal difference)
        runTest("Test 6: Full table scan (COUNT(*))",
                "SELECT COUNT(*) FROM mysql_import.historico",
                "SELECT COUNT(*) FROM mysql_import.historico_mt");
        
        // Test 7: Sampling query (MergeTree supports sampling)
        runTest("Test 7: Top modes by frequency",
                "SELECT modo, COUNT(*) as cnt FROM mysql_import.historico GROUP BY modo ORDER BY cnt DESC LIMIT 10",
                "SELECT modo, COUNT(*) as cnt FROM mysql_import.historico_mt GROUP BY modo ORDER BY cnt DESC LIMIT 10");
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "- Large improvements expected for filtered queries (using primary key)" << std::endl;
        std::cout << "- Minimal improvements for full table scans" << std::endl;
        std::cout << "- MergeTree excels at range queries and point lookups" << std::endl;
    }
};

int main() {
    PerformanceTester tester;
    
    if (!tester.loadChdbLibrary()) {
        return 1;
    }
    
    tester.runAllTests();
    
    return 0;
}