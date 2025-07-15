/**********************************************************************
 *  historico_feeder.cpp  - MergeTree loader with row-based chunking
 **********************************************************************/

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
#include <dlfcn.h>
#include <cstring>
#include <chrono>
#include <memory>
#include <iomanip>
#include <algorithm>
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

class MergeTreeLoader {
private:
    MYSQL* mysql_conn;
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    MergeTreeLoader() : mysql_conn(nullptr), chdb_handle(nullptr) {}
    
    ~MergeTreeLoader() {
        if (mysql_conn) {
            mysql_close(mysql_conn);
        }
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool loadChdbLibrary() {
        chdb_handle = dlopen("/home/cslog/chdb/libchdb.so", RTLD_LAZY);
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
    
    bool connectToMySQL(const std::string& host, const std::string& user, 
                       const std::string& password, const std::string& database) {
        mysql_conn = mysql_init(nullptr);
        if (!mysql_conn) {
            std::cerr << "MySQL init failed" << std::endl;
            return false;
        }
        
        if (!mysql_real_connect(mysql_conn, host.c_str(), user.c_str(), 
                               password.c_str(), database.c_str(), 0, nullptr, 0)) {
            std::cerr << "MySQL connection failed: " << mysql_error(mysql_conn) << std::endl;
            return false;
        }
        
        std::cout << "[" << getCurrentTimestamp() << "] Connected to MySQL successfully!" << std::endl;
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
    
    std::string escapeString(const std::string& str) {
        std::string escaped;
        for (char c : str) {
            if (c == '\'') {
                escaped += "\\'";
            } else if (c == '\\') {
                escaped += "\\\\";
            } else {
                escaped += c;
            }
        }
        return escaped;
    }

    void createTables(bool skip_texto = false) {
        std::cout << "\n[" << getCurrentTimestamp() << "] Creating tables with MergeTree engine..." << std::endl;
        
        auto result = executeQuery("CREATE DATABASE IF NOT EXISTS mysql_import");
        if (result) {
            free_result_v2(result);
        }
        
        // Create HISTORICO table with MergeTree
        result = executeQuery(
            "CREATE TABLE IF NOT EXISTS mysql_import.historico ("
            "id_contr Int32, seq UInt16, id_funcionario Int32, "
            "id_tel Int32, data DateTime, codigo UInt16, modo String"
            ") ENGINE = MergeTree() ORDER BY (id_contr, seq)"
        );
        
        if (result) {
            if (result->error_message) {
                std::cout << "Create HISTORICO table error: " << result->error_message << std::endl;
                free_result_v2(result);
                return;
            }
            std::cout << "HISTORICO table created successfully" << std::endl;
            free_result_v2(result);
        }
        
        // Create HISTORICO_TEXTO table only if not skipping
        if (!skip_texto) {
            result = executeQuery(
                "CREATE TABLE IF NOT EXISTS mysql_import.historico_texto ("
                "id_contr Int32, seq UInt16, mensagem String, "
                "motivo String, autorizacao String"
                ") ENGINE = MergeTree() ORDER BY (id_contr, seq)"
            );
            
            if (result) {
                if (result->error_message) {
                    std::cout << "Create HISTORICO_TEXTO table error: " << result->error_message << std::endl;
                    free_result_v2(result);
                    return;
                }
                std::cout << "HISTORICO_TEXTO table created successfully" << std::endl;
                free_result_v2(result);
            }
        }
    }
    
    void loadHistoricoMergeTree(bool skip_texto = false, long long provided_row_count = 0, long long start_offset = 0) {
        createTables(skip_texto);
        
        // Get total row count if not provided
        long long total_rows = provided_row_count;
        
        if (total_rows == 0) {
            std::cout << "\n[" << getCurrentTimestamp() << "] Getting row count from HISTORICO table..." << std::endl;
            
            if (mysql_query(mysql_conn, "SELECT COUNT(*) FROM HISTORICO")) {
                std::cerr << "Failed to get row count: " << mysql_error(mysql_conn) << std::endl;
                return;
            }
            
            MYSQL_RES* count_result = mysql_store_result(mysql_conn);
            if (!count_result) {
                std::cerr << "Failed to store count result: " << mysql_error(mysql_conn) << std::endl;
                return;
            }
            
            MYSQL_ROW row = mysql_fetch_row(count_result);
            if (row && row[0]) {
                total_rows = std::stoll(row[0]);
            }
            mysql_free_result(count_result);
        } else {
            std::cout << "\n[" << getCurrentTimestamp() << "] Using provided row count: " << total_rows << std::endl;
        }
        
        if (total_rows == 0) {
            std::cout << "No data found in HISTORICO table" << std::endl;
            return;
        }
        
        std::cout << "Total rows to process: " << total_rows << std::endl;
        if (start_offset > 0) {
            std::cout << "Starting from offset: " << start_offset << std::endl;
        }
        
        // Load data in chunks using row-based approach
        const int ROWS_PER_CHUNK = 50000;  // Process 50k rows at a time
        const int BATCH_SIZE = 1000;       // Insert 1000 rows per batch for MergeTree
        
        long long offset = start_offset;
        long long total_historico_loaded = start_offset;  // Start from offset when resuming
        int chunk_number = start_offset / ROWS_PER_CHUNK;  // Calculate starting chunk number
        auto start_time = std::chrono::steady_clock::now();
        
        int total_chunks = (total_rows + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
        
        while (offset < total_rows) {
            chunk_number++;
            auto chunk_start_time = std::chrono::steady_clock::now();
            
            std::cout << "\n[" << getCurrentTimestamp() << "] Processing chunk " << chunk_number << "/" << total_chunks 
                      << " (rows " << offset << "-" << std::min(offset + ROWS_PER_CHUNK, total_rows) << " of " << total_rows << ")..." << std::endl;
            std::cout.flush();
            
            std::stringstream query;
            query << "SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO "
                  << "FROM HISTORICO "
                  << "ORDER BY ID_CONTR, SEQ "
                  << "LIMIT " << ROWS_PER_CHUNK << " OFFSET " << offset;
            
            if (mysql_query(mysql_conn, query.str().c_str())) {
                std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
                offset += ROWS_PER_CHUNK;
                continue;
            }
            
            // Use mysql_use_result for streaming to reduce memory usage
            MYSQL_RES* mysql_result = mysql_use_result(mysql_conn);
            if (!mysql_result) {
                std::cerr << "MySQL use result failed: " << mysql_error(mysql_conn) << std::endl;
                offset += ROWS_PER_CHUNK;
                continue;
            }
            
            MYSQL_ROW row;
            int batch_count = 0;
            int chunk_rows = 0;
            std::stringstream batch_insert;
            batch_insert << "INSERT INTO mysql_import.historico VALUES ";
            bool first = true;
            
            while ((row = mysql_fetch_row(mysql_result))) {
                if (!first) batch_insert << ", ";
                first = false;
                
                batch_insert << "("
                            << (row[0] ? row[0] : "0") << ", "
                            << (row[1] ? row[1] : "0") << ", "
                            << (row[2] ? row[2] : "0") << ", "
                            << (row[3] ? row[3] : "0") << ", "
                            << "'" << (row[4] ? row[4] : "1970-01-01 00:00:00") << "', "
                            << (row[5] ? row[5] : "0") << ", "
                            << "'" << (row[6] ? escapeString(row[6]) : "*") << "')";
                batch_count++;
                chunk_rows++;
                
                // Insert in batches for better performance
                if (batch_count == BATCH_SIZE) {
                    auto ch_result = executeQuery(batch_insert.str());
                    if (ch_result) {
                        if (ch_result->error_message) {
                            std::cerr << "Batch insert error: " << ch_result->error_message << std::endl;
                        }
                        free_result_v2(ch_result);
                    }
                    total_historico_loaded += batch_count;
                    batch_count = 0;
                    batch_insert.str("");
                    batch_insert << "INSERT INTO mysql_import.historico VALUES ";
                    first = true;
                    
                    if (total_historico_loaded % 10000 == 0) {
                        auto current_time = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                        long long rows_processed_this_session = total_historico_loaded - start_offset;
                        std::cout << "  Progress: " << total_historico_loaded << " HISTORICO rows loaded ("
                                  << (elapsed > 0 ? rows_processed_this_session / elapsed : rows_processed_this_session) << " rows/sec)" << std::endl;
                    }
                }
            }
            
            // Insert remaining rows
            if (!first && batch_count > 0) {
                auto ch_result = executeQuery(batch_insert.str());
                if (ch_result) {
                    if (ch_result->error_message) {
                        std::cerr << "Final batch insert error: " << ch_result->error_message << std::endl;
                    }
                    free_result_v2(ch_result);
                }
                total_historico_loaded += batch_count;
            }
            
            mysql_free_result(mysql_result);
            std::cout << "  HISTORICO: " << chunk_rows << " rows loaded for this chunk" << std::endl;
            
            // Calculate and display chunk processing time
            auto chunk_end_time = std::chrono::steady_clock::now();
            auto chunk_duration = std::chrono::duration_cast<std::chrono::seconds>(chunk_end_time - chunk_start_time).count();
            
            auto elapsed_total = std::chrono::duration_cast<std::chrono::seconds>(chunk_end_time - start_time).count();
            long long rows_processed_this_session = total_historico_loaded - start_offset;
            long long rows_per_sec = elapsed_total > 0 ? rows_processed_this_session / elapsed_total : 0;
            
            std::cout << "  [" << getCurrentTimestamp() << "] Chunk " << chunk_number 
                      << " completed in " << chunk_duration << " seconds"
                      << " (avg: " << rows_per_sec << " rows/sec)" << std::endl;
            
            offset += ROWS_PER_CHUNK;
            
            // Progress estimate
            if (total_rows > 0) {
                double progress = (double)offset / total_rows * 100;
                long long remaining_rows = total_rows - offset;
                long long eta_seconds = rows_per_sec > 0 ? remaining_rows / rows_per_sec : 0;
                
                std::cout << "  Progress: " << std::fixed << std::setprecision(1) << progress << "% "
                          << "(" << total_historico_loaded << "/" << total_rows << " rows) "
                          << "- ETA: " << (eta_seconds / 60) << " minutes" << std::endl;
            }
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "\n[" << getCurrentTimestamp() << "] Loading completed!" << std::endl;
        long long rows_processed_this_session = total_historico_loaded - start_offset;
        std::cout << "Total HISTORICO rows loaded: " << total_historico_loaded << std::endl;
        std::cout << "Rows processed this session: " << rows_processed_this_session << " in " << duration << " seconds" << std::endl;
        std::cout << "Average speed: " << (duration > 0 ? rows_processed_this_session / duration : rows_processed_this_session) << " rows/second" << std::endl;
        
        // Verify table
        std::cout << "\nVerifying loaded data:" << std::endl;
        auto result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico");
        if (result && result->buf) {
            std::cout << "HISTORICO count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        if (!skip_texto) {
            result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto");
            if (result && result->buf) {
                std::cout << "HISTORICO_TEXTO count: " << result->buf << std::endl;
                free_result_v2(result);
            }
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <host> <user> <password> <database> [options]" << std::endl;
        std::cout << "\nThis version uses MergeTree engine with row-based chunking" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --skip-texto           Skip loading HISTORICO_TEXTO table" << std::endl;
        std::cout << "  --row-count <count>    Provide row count to skip COUNT(*) query" << std::endl;
        std::cout << "  --offset <offset>      Start from this row offset (default: 0)" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " host user pass db --skip-texto" << std::endl;
        std::cout << "  " << argv[0] << " host user pass db --row-count 300266692" << std::endl;
        std::cout << "  " << argv[0] << " host user pass db --row-count 300266692 --offset 1000000" << std::endl;
        return 1;
    }
    
    bool skip_texto = false;
    long long provided_row_count = 0;
    long long start_offset = 0;
    
    // Parse command line options
    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--skip-texto") {
            skip_texto = true;
        } else if (arg == "--row-count" && i + 1 < argc) {
            provided_row_count = std::stoll(argv[++i]);
        } else if (arg == "--offset" && i + 1 < argc) {
            start_offset = std::stoll(argv[++i]);
        }
    }
    
    MergeTreeLoader loader;
    std::cout << "[" << loader.getCurrentTimestamp() << "] Starting historico_feeder..." << std::endl;
    if (skip_texto) {
        std::cout << "[" << loader.getCurrentTimestamp() << "] Skipping HISTORICO_TEXTO table" << std::endl;
    }
    
    if (!loader.loadChdbLibrary()) {
        return 1;
    }
    
    if (!loader.connectToMySQL(argv[1], argv[2], argv[3], argv[4])) {
        return 1;
    }
    
    // Load data using MergeTree engine
    loader.loadHistoricoMergeTree(skip_texto, provided_row_count, start_offset);
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}