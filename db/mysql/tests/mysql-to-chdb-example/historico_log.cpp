#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
#include <dlfcn.h>
#include <cstring>
#include <chrono>
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

class LogTableLoader {
private:
    MYSQL* mysql_conn;
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    LogTableLoader() : mysql_conn(nullptr), chdb_handle(nullptr) {}
    
    ~LogTableLoader() {
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
        
        std::cout << "Connected to MySQL successfully!" << std::endl;
        return true;
    }
    
    struct local_result_v2* executeQuery(const std::string& query) {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        // Add multiquery support for batch operations
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        return query_stable_v2(argv.size(), argv.data());
    }
    
    void testBasicQuery() {
        std::cout << "\nTesting basic query..." << std::endl;
        auto result = executeQuery("SELECT 1+1");
        if (result) {
            if (result->buf) {
                std::cout << "Result: " << result->buf << std::endl;
            }
            if (result->error_message) {
                std::cout << "Error: " << result->error_message << std::endl;
            }
            free_result_v2(result);
        }
    }
    
    void createSimpleTable() {
        std::cout << "\nCreating simple database..." << std::endl;
        
        // Try each command separately
        auto result = executeQuery("CREATE DATABASE IF NOT EXISTS test_db");
        if (result) {
            if (result->error_message) {
                std::cout << "Create DB error: " << result->error_message << std::endl;
            } else {
                std::cout << "Database created" << std::endl;
            }
            free_result_v2(result);
        }
        
        std::cout << "\nCreating simple table..." << std::endl;
        
        // Use Memory engine instead of MergeTree
        result = executeQuery("CREATE TABLE IF NOT EXISTS test_db.simple (id Int32, value String) ENGINE = Memory");
        if (result) {
            if (result->error_message) {
                std::cout << "Create table error: " << result->error_message << std::endl;
            } else {
                std::cout << "Table created" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Test insert
        std::cout << "\nTesting insert..." << std::endl;
        result = executeQuery("INSERT INTO test_db.simple VALUES (1, 'test')");
        if (result) {
            if (result->error_message) {
                std::cout << "Insert error: " << result->error_message << std::endl;
            } else {
                std::cout << "Insert successful" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Test select
        result = executeQuery("SELECT * FROM test_db.simple");
        if (result) {
            if (result->buf) {
                std::cout << "Data: " << result->buf << std::endl;
            }
            free_result_v2(result);
        }
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

    void createTables() {
        std::cout << "\nCreating tables with Log engine..." << std::endl;
        
        auto result = executeQuery("CREATE DATABASE IF NOT EXISTS mysql_import");
        if (result) {
            free_result_v2(result);
        }
        
        // Create HISTORICO table
        result = executeQuery(
            "CREATE TABLE IF NOT EXISTS mysql_import.historico ("
            "id_contr Int32, seq UInt16, id_funcionario Int32, "
            "id_tel Int32, data DateTime, codigo UInt16, modo String"
            ") ENGINE = Log"
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
        
        // Create HISTORICO_TEXTO table
        result = executeQuery(
            "CREATE TABLE IF NOT EXISTS mysql_import.historico_texto ("
            "id_contr Int32, seq UInt16, mensagem String, "
            "motivo String, autorizacao String"
            ") ENGINE = Log"
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
    
    int loadHistoricoTextoChunk(int min_id, int max_id) {
        std::stringstream query;
        query << "SELECT ID_CONTR, SEQ, MENSAGEM, MOTIVO, AUTORIZACAO "
              << "FROM HISTORICO_TEXTO WHERE ID_CONTR >= " << min_id 
              << " AND ID_CONTR < " << max_id
              << " ORDER BY ID_CONTR, SEQ";
        
        if (mysql_query(mysql_conn, query.str().c_str())) {
            std::cerr << "  MySQL query failed for HISTORICO_TEXTO: " << mysql_error(mysql_conn) << std::endl;
            return 0;
        }
        
        MYSQL_RES* mysql_result = mysql_store_result(mysql_conn);
        if (!mysql_result) {
            std::cerr << "  MySQL store result failed for HISTORICO_TEXTO: " << mysql_error(mysql_conn) << std::endl;
            return 0;
        }
        
        MYSQL_ROW row;
        int batch_count = 0;
        int texto_rows = 0;
        std::stringstream batch_insert;
        batch_insert << "INSERT INTO mysql_import.historico_texto VALUES ";
        bool first = true;
        
        while ((row = mysql_fetch_row(mysql_result))) {
            if (!first) batch_insert << ", ";
            first = false;
            
            batch_insert << "("
                        << (row[0] ? row[0] : "0") << ", "
                        << (row[1] ? row[1] : "0") << ", "
                        << "'" << (row[2] ? escapeString(row[2]) : "") << "', "
                        << "'" << (row[3] ? escapeString(row[3]) : "") << "', "
                        << "'" << (row[4] ? escapeString(row[4]) : "") << "')";
            batch_count++;
            texto_rows++;
            
            // Insert in batches of 100 rows (smaller batch for text data)
            if (batch_count == 100) {
                auto ch_result = executeQuery(batch_insert.str());
                if (ch_result) {
                    if (ch_result->error_message) {
                        std::cerr << "  HISTORICO_TEXTO batch insert error: " << ch_result->error_message << std::endl;
                    }
                    free_result_v2(ch_result);
                }
                batch_count = 0;
                batch_insert.str("");
                batch_insert << "INSERT INTO mysql_import.historico_texto VALUES ";
                first = true;
            }
        }
        
        // Insert remaining rows
        if (!first && batch_count > 0) {
            auto ch_result = executeQuery(batch_insert.str());
            if (ch_result) {
                if (ch_result->error_message) {
                    std::cerr << "  HISTORICO_TEXTO final batch insert error: " << ch_result->error_message << std::endl;
                }
                free_result_v2(ch_result);
            }
        }
        
        mysql_free_result(mysql_result);
        std::cout << "  HISTORICO_TEXTO: " << texto_rows << " rows loaded for this chunk" << std::endl;
        return texto_rows;
    }

    void loadHistoricoLog() {
        createTables();
        
        // Get min and max ID_CONTR values to process all data
        std::cout << "\nGetting ID range from HISTORICO table..." << std::endl;
        int min_id = 0, max_id = 0;
        
        if (mysql_query(mysql_conn, "SELECT MIN(ID_CONTR), MAX(ID_CONTR) FROM HISTORICO")) {
            std::cerr << "Failed to get ID range: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        MYSQL_RES* range_result = mysql_store_result(mysql_conn);
        if (range_result) {
            MYSQL_ROW row = mysql_fetch_row(range_result);
            if (row && row[0] && row[1]) {
                min_id = std::stoi(row[0]);
                max_id = std::stoi(row[1]) + 1; // +1 to include the last ID
            }
            mysql_free_result(range_result);
        }
        
        if (min_id == 0 && max_id == 0) {
            std::cout << "No data found in HISTORICO table" << std::endl;
            return;
        }
        
        std::cout << "ID_CONTR range: [" << min_id << ", " << max_id << ")" << std::endl;
        
        // Load data in chunks
        const int CHUNK_SIZE = 10000;
        int total_historico_loaded = 0;
        int total_texto_loaded = 0;
        int chunk_number = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        int total_chunks = (max_id - min_id + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        for (int current_id = min_id; current_id < max_id; current_id += CHUNK_SIZE) {
            int chunk_end = std::min(current_id + CHUNK_SIZE, max_id);
            chunk_number++;
            
            std::cout << "\nProcessing chunk " << chunk_number << "/" << total_chunks 
                      << " (ID range: " << current_id << "-" << chunk_end << ")..." << std::endl;
            std::cout.flush();
            
            std::stringstream query;
            query << "SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO "
                  << "FROM HISTORICO WHERE ID_CONTR >= " << current_id 
                  << " AND ID_CONTR < " << chunk_end
                  << " ORDER BY ID_CONTR, SEQ";
            
            std::cout << "  Executing MySQL query..." << std::endl;
            if (mysql_query(mysql_conn, query.str().c_str())) {
                std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
                continue;
            }
            
            std::cout << "  Storing result..." << std::endl;
            MYSQL_RES* mysql_result = mysql_store_result(mysql_conn);
            if (!mysql_result) {
                std::cerr << "MySQL store result failed: " << mysql_error(mysql_conn) << std::endl;
                continue;
            }
            std::cout << "  Processing rows..." << std::endl;
            
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
                
                // Insert in batches of 500 rows for better stability
                if (batch_count == 500) {
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
                        std::cout << "  Progress: " << total_historico_loaded << " HISTORICO rows loaded ("
                                  << (elapsed > 0 ? total_historico_loaded / elapsed : total_historico_loaded) << " rows/sec)" << std::endl;
                    }
                }
            }
            
            // Insert remaining rows
            if (!first && batch_count > 0) {
                std::cout << "  Inserting final " << batch_count << " rows of chunk..." << std::endl;
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
            
            // Now load corresponding HISTORICO_TEXTO rows for this chunk
            std::cout << "  Loading HISTORICO_TEXTO for this chunk..." << std::endl;
            int texto_count = loadHistoricoTextoChunk(current_id, chunk_end);
            total_texto_loaded += texto_count;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "\nTotal HISTORICO rows loaded: " << total_historico_loaded << " in " << duration << " seconds" << std::endl;
        std::cout << "Average speed: " << (duration > 0 ? total_historico_loaded / duration : total_historico_loaded) << " rows/second" << std::endl;
        
        // Verify both tables
        std::cout << "\nVerifying loaded data:" << std::endl;
        auto result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico");
        if (result && result->buf) {
            std::cout << "HISTORICO count: " << result->buf << std::endl;
            free_result_v2(result);
        }
        
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto");
        if (result && result->buf) {
            std::cout << "HISTORICO_TEXTO count: " << result->buf << std::endl;
            free_result_v2(result);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <host> <user> <password> <database>" << std::endl;
        std::cout << "\nThis version uses Log engine for better compatibility with chdb" << std::endl;
        std::cout << "After loading, use convert_to_mergetree for better query performance" << std::endl;
        return 1;
    }
    
    LogTableLoader loader;
    
    if (!loader.loadChdbLibrary()) {
        return 1;
    }
    
    // Test basic functionality first
    loader.testBasicQuery();
    loader.createSimpleTable();
    
    if (!loader.connectToMySQL(argv[1], argv[2], argv[3], argv[4])) {
        return 1;
    }
    
    // Load data using Log engine
    loader.loadHistoricoLog();
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}