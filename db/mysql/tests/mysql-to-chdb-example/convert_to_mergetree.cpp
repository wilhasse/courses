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

class TableConverter {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    TableConverter() : chdb_handle(nullptr) {}
    
    ~TableConverter() {
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
        
        std::cout << "chdb library loaded successfully" << std::endl;
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
    
    void convertToMergeTree() {
        std::cout << "\nConverting Log tables to MergeTree for better query performance..." << std::endl;
        
        // First, check the current row counts
        std::cout << "\nChecking current table sizes:" << std::endl;
        auto result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico");
        if (result && result->buf) {
            std::cout << "HISTORICO row count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto");
        if (result && result->buf) {
            std::cout << "HISTORICO_TEXTO row count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        std::cout << "\nCreating MergeTree tables..." << std::endl;
        
        // Create HISTORICO MergeTree table
        result = executeQuery(
            "CREATE TABLE IF NOT EXISTS mysql_import.historico_mt ("
            "id_contr Int32, seq UInt16, id_funcionario Int32, "
            "id_tel Int32, data DateTime, codigo UInt16, modo String"
            ") ENGINE = MergeTree() "
            "ORDER BY (id_contr, seq) "
            "SETTINGS index_granularity = 8192"
        );
        
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating HISTORICO MergeTree table: " << result->error_message << std::endl;
                free_result_v2(result);
                return;
            }
            std::cout << "HISTORICO MergeTree table created successfully" << std::endl;
            free_result_v2(result);
        }
        
        // Create HISTORICO_TEXTO MergeTree table
        result = executeQuery(
            "CREATE TABLE IF NOT EXISTS mysql_import.historico_texto_mt ("
            "id_contr Int32, seq UInt16, mensagem String, "
            "motivo String, autorizacao String"
            ") ENGINE = MergeTree() "
            "ORDER BY (id_contr, seq) "
            "SETTINGS index_granularity = 8192"
        );
        
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating HISTORICO_TEXTO MergeTree table: " << result->error_message << std::endl;
                free_result_v2(result);
                return;
            }
            std::cout << "HISTORICO_TEXTO MergeTree table created successfully" << std::endl;
            free_result_v2(result);
        }
        
        std::cout << "\nCopying data from Log to MergeTree tables..." << std::endl;
        
        // Copy HISTORICO data
        std::cout << "Copying HISTORICO data..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        result = executeQuery("INSERT INTO mysql_import.historico_mt SELECT * FROM mysql_import.historico");
        
        if (result) {
            if (result->error_message) {
                std::cerr << "Error copying HISTORICO data: " << result->error_message << std::endl;
            } else {
                std::cout << "HISTORICO data copied successfully" << std::endl;
            }
            free_result_v2(result);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "HISTORICO copy completed in " << duration << " seconds" << std::endl;
        
        // Copy HISTORICO_TEXTO data
        std::cout << "\nCopying HISTORICO_TEXTO data..." << std::endl;
        start_time = std::chrono::steady_clock::now();
        
        result = executeQuery("INSERT INTO mysql_import.historico_texto_mt SELECT * FROM mysql_import.historico_texto");
        
        if (result) {
            if (result->error_message) {
                std::cerr << "Error copying HISTORICO_TEXTO data: " << result->error_message << std::endl;
            } else {
                std::cout << "HISTORICO_TEXTO data copied successfully" << std::endl;
            }
            free_result_v2(result);
        }
        
        end_time = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "HISTORICO_TEXTO copy completed in " << duration << " seconds" << std::endl;
        
        // Verify the copies
        std::cout << "\nVerifying copied data:" << std::endl;
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_mt");
        if (result && result->buf) {
            std::cout << "HISTORICO MergeTree row count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto_mt");
        if (result && result->buf) {
            std::cout << "HISTORICO_TEXTO MergeTree row count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        // Compare query performance
        std::cout << "\n=== Query Performance Comparison ===" << std::endl;
        
        // Test 1: Count with filter
        std::cout << "\nTest 1: COUNT with filter on id_contr" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico WHERE id_contr BETWEEN 1000000 AND 2000000");
        auto log_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "Log result: " << result->buf;
            free_result_v2(result);
        }
        auto log_duration = std::chrono::duration_cast<std::chrono::milliseconds>(log_time - start_time).count();
        std::cout << " (Time: " << log_duration << "ms)" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_mt WHERE id_contr BETWEEN 1000000 AND 2000000");
        auto mt_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "MergeTree result: " << result->buf;
            free_result_v2(result);
        }
        auto mt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mt_time - start_time).count();
        std::cout << " (Time: " << mt_duration << "ms)" << std::endl;
        
        std::cout << "Speed improvement: " << (log_duration > 0 ? (float)log_duration / mt_duration : 0) << "x faster" << std::endl;
        
        // Test 2: Aggregation
        std::cout << "\nTest 2: Aggregation by codigo" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico WHERE id_contr < 100000 GROUP BY codigo ORDER BY cnt DESC LIMIT 5");
        log_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "Log result:\n" << result->buf;
            free_result_v2(result);
        }
        log_duration = std::chrono::duration_cast<std::chrono::milliseconds>(log_time - start_time).count();
        std::cout << "Time: " << log_duration << "ms" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico_mt WHERE id_contr < 100000 GROUP BY codigo ORDER BY cnt DESC LIMIT 5");
        mt_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "\nMergeTree result:\n" << result->buf;
            free_result_v2(result);
        }
        mt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mt_time - start_time).count();
        std::cout << "Time: " << mt_duration << "ms" << std::endl;
        
        std::cout << "Speed improvement: " << (log_duration > 0 ? (float)log_duration / mt_duration : 0) << "x faster" << std::endl;
        
        // Test 3: HISTORICO_TEXTO text search
        std::cout << "\nTest 3: Text search in HISTORICO_TEXTO" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto WHERE mensagem LIKE '%erro%' OR motivo LIKE '%falha%'");
        log_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "Log result: " << result->buf;
            free_result_v2(result);
        }
        log_duration = std::chrono::duration_cast<std::chrono::milliseconds>(log_time - start_time).count();
        std::cout << " (Time: " << log_duration << "ms)" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto_mt WHERE mensagem LIKE '%erro%' OR motivo LIKE '%falha%'");
        mt_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "MergeTree result: " << result->buf;
            free_result_v2(result);
        }
        mt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mt_time - start_time).count();
        std::cout << " (Time: " << mt_duration << "ms)" << std::endl;
        
        std::cout << "Speed improvement: " << (log_duration > 0 ? (float)log_duration / mt_duration : 0) << "x faster" << std::endl;
        
        // Test 4: Join between tables
        std::cout << "\nTest 4: Join query between HISTORICO and HISTORICO_TEXTO" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico h INNER JOIN mysql_import.historico_texto t ON h.id_contr = t.id_contr AND h.seq = t.seq WHERE h.codigo = 51 AND h.id_contr < 100000");
        log_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "Log result: " << result->buf;
            free_result_v2(result);
        }
        log_duration = std::chrono::duration_cast<std::chrono::milliseconds>(log_time - start_time).count();
        std::cout << " (Time: " << log_duration << "ms)" << std::endl;
        
        start_time = std::chrono::steady_clock::now();
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_mt h INNER JOIN mysql_import.historico_texto_mt t ON h.id_contr = t.id_contr AND h.seq = t.seq WHERE h.codigo = 51 AND h.id_contr < 100000");
        mt_time = std::chrono::steady_clock::now();
        if (result) {
            if (result->buf) std::cout << "MergeTree result: " << result->buf;
            free_result_v2(result);
        }
        mt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mt_time - start_time).count();
        std::cout << " (Time: " << mt_duration << "ms)" << std::endl;
        
        std::cout << "Speed improvement: " << (log_duration > 0 ? (float)log_duration / mt_duration : 0) << "x faster" << std::endl;
        
        std::cout << "\nConversion complete! Use '_mt' suffix tables for better query performance." << std::endl;
        std::cout << "- historico_mt for HISTORICO queries" << std::endl;
        std::cout << "- historico_texto_mt for HISTORICO_TEXTO queries" << std::endl;
    }
};

int main() {
    TableConverter converter;
    
    if (!converter.loadChdbLibrary()) {
        return 1;
    }
    
    converter.convertToMergeTree();
    
    return 0;
}