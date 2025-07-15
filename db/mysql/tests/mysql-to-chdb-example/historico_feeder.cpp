#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
#include <dlfcn.h>
#include <cstring>
#include <limits>
#include <chrono>
#include <thread>
#include "common.h"

// Use the deprecated but stable v2 API
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

struct Historico {
    int id_contr;
    int seq;
    int id_funcionario;
    int id_tel;
    std::string data;
    int codigo;
    std::string modo;
};

struct HistoricoTexto {
    int id_contr;
    int seq;
    std::string mensagem;
    std::string motivo;
    std::string autorizacao;
};

class HistoricoFeeder {
private:
    MYSQL* mysql_conn;
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    const int CHUNK_SIZE = 1000;
    bool test_mode = false;
    int test_limit = 100;
    int total_operations = 0;
    const int MAX_OPERATIONS = 500;  // Restart chdb after this many operations
    
public:
    HistoricoFeeder() : mysql_conn(nullptr), chdb_handle(nullptr) {}
    
    void setTestMode(bool mode, int limit = 100) {
        test_mode = mode;
        test_limit = limit;
    }
    
    ~HistoricoFeeder() {
        if (mysql_conn) {
            mysql_close(mysql_conn);
        }
        if (chdb_handle) {
            dlclose(chdb_handle);
        }
    }
    
    bool loadChdbLibrary() {
        const char* lib_paths[] = {
            "/home/cslog/chdb/libchdb.so",
            "./libchdb.so",
            "libchdb.so"
        };
        
        for (const char* path : lib_paths) {
            chdb_handle = dlopen(path, RTLD_LAZY);
            if (chdb_handle) {
                std::cout << "Loaded chdb library from: " << path << std::endl;
                break;
            }
        }
        
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
        
        std::cout << "Connected to MySQL server: " << user << "@" << host << "/" << database << std::endl;
        return true;
    }
    
    bool reloadChdbLibrary() {
        // Close existing handle
        if (chdb_handle) {
            dlclose(chdb_handle);
            chdb_handle = nullptr;
        }
        
        // Small delay before reloading
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        return loadChdbLibrary();
    }
    
    struct local_result_v2* executeQuery(const std::string& query, const std::string& output_format = "CSV") {
        // Check if we need to restart chdb
        if (total_operations >= MAX_OPERATIONS) {
            std::cout << "\n    Restarting chdb library (reached " << total_operations << " operations)..." << std::endl;
            if (!reloadChdbLibrary()) {
                std::cerr << "Failed to reload chdb library!" << std::endl;
                return nullptr;
            }
            total_operations = 0;
            std::cout << "    chdb library restarted successfully" << std::endl;
        }
        
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--output-format=" + output_format);
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        total_operations++;
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
    
    void createTables() {
        std::cout << "Creating ClickHouse tables..." << std::endl;
        
        // Create database
        auto result = executeQuery("CREATE DATABASE IF NOT EXISTS mysql_import");
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating database: " << result->error_message << std::endl;
            } else {
                std::cout << "Database created/verified" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Create HISTORICO table
        std::string create_historico = R"(
            CREATE TABLE IF NOT EXISTS mysql_import.historico (
                id_contr Int32,
                seq UInt16,
                id_funcionario Int32,
                id_tel Int32,
                data DateTime,
                codigo UInt16,
                modo String
            ) ENGINE = MergeTree()
            PRIMARY KEY (id_contr, seq)
            ORDER BY (id_contr, seq)
        )";
        
        result = executeQuery(create_historico);
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating historico table: " << result->error_message << std::endl;
            } else {
                std::cout << "Historico table created" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Create HISTORICO_TEXTO table
        std::string create_historico_texto = R"(
            CREATE TABLE IF NOT EXISTS mysql_import.historico_texto (
                id_contr Int32,
                seq UInt16,
                mensagem String,
                motivo String,
                autorizacao String
            ) ENGINE = MergeTree()
            PRIMARY KEY (id_contr, seq)
            ORDER BY (id_contr, seq)
        )";
        
        result = executeQuery(create_historico_texto);
        if (result) {
            if (result->error_message) {
                std::cerr << "Error creating historico_texto table: " << result->error_message << std::endl;
            } else {
                std::cout << "Historico_texto table created" << std::endl;
            }
            free_result_v2(result);
        }
        
        // Clear existing data
        executeQuery("TRUNCATE TABLE IF EXISTS mysql_import.historico");
        executeQuery("TRUNCATE TABLE IF EXISTS mysql_import.historico_texto");
    }
    
    void loadHistoricoChunk(int min_id, int max_id) {
        std::stringstream query;
        query << "SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO "
              << "FROM HISTORICO WHERE ID_CONTR >= " << min_id 
              << " AND ID_CONTR < " << max_id
              << " ORDER BY ID_CONTR, SEQ";
        
        std::cout << "  Executing MySQL query for HISTORICO..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        if (mysql_query(mysql_conn, query.str().c_str())) {
            std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        std::cout << "  Fetching results..." << std::endl;
        MYSQL_RES* result = mysql_store_result(mysql_conn);
        if (!result) {
            std::cerr << "MySQL store result failed: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        MYSQL_ROW row;
        int count = 0;
        int batch_count = 0;
        std::stringstream batch_insert;
        batch_insert << "INSERT INTO mysql_import.historico VALUES ";
        bool first = true;
        
        while ((row = mysql_fetch_row(result))) {
            if (test_mode && count >= test_limit) {
                std::cout << "    Test mode: Stopping at " << count << " records" << std::endl;
                break;
            }
            
            if (!first) batch_insert << ", ";
            first = false;
            
            // Debug problematic rows
            if (count >= 250 && count <= 270) {
                std::cout << "\n    DEBUG Row " << count + 1 << ": ID_CONTR=" << (row[0] ? row[0] : "NULL") 
                         << ", SEQ=" << (row[1] ? row[1] : "NULL") << std::endl;
            }
            
            batch_insert << "("
                        << (row[0] ? row[0] : "0") << ", "
                        << (row[1] ? row[1] : "0") << ", "
                        << (row[2] ? row[2] : "0") << ", "
                        << (row[3] ? row[3] : "0") << ", "
                        << "'" << (row[4] ? row[4] : "1970-01-01 00:00:00") << "', "
                        << (row[5] ? row[5] : "0") << ", "
                        << "'" << (row[6] ? escapeString(row[6]) : "*") << "')";
            count++;
            
            // Insert in batches of 10 rows (reduced from 100)
            if (count % 10 == 0) {
                std::cout << "    Preparing to insert batch at row " << count << "..." << std::flush;
                
                // Save query for debugging
                std::string query = batch_insert.str();
                
                try {
                    auto ch_result = executeQuery(query);
                    if (ch_result) {
                        if (ch_result->error_message) {
                            std::cerr << "\n    ClickHouse insert error: " << ch_result->error_message << std::endl;
                            std::cerr << "    Failed query: " << query.substr(0, 200) << "..." << std::endl;
                            // Skip this batch and continue
                            batch_insert.str("");
                            batch_insert << "INSERT INTO mysql_import.historico VALUES ";
                            first = true;
                            free_result_v2(ch_result);  // Always free the result
                            continue;
                        } else {
                            std::cout << " OK" << std::endl;
                        }
                        free_result_v2(ch_result);
                        
                        // Add a small delay every 10 batches to avoid overwhelming the system
                        if (batch_count > 0 && batch_count % 10 == 0) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        }
                    } else {
                        std::cerr << "\n    ClickHouse query returned null result" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "\n    Exception during insert: " << e.what() << std::endl;
                }
                
                batch_count++;
                if (batch_count % 100 == 0) {
                    std::cout << "    >>> Total inserted: " << count << " rows" << std::endl;
                }
                batch_insert.str("");
                batch_insert << "INSERT INTO mysql_import.historico VALUES ";
                first = true;
            }
        }
        
        // Insert remaining rows
        if (!first) {
            auto ch_result = executeQuery(batch_insert.str());
            if (ch_result) {
                if (ch_result->error_message) {
                    std::cerr << "    ClickHouse insert error: " << ch_result->error_message << std::endl;
                }
                free_result_v2(ch_result);
            }
        }
        
        mysql_free_result(result);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "  Loaded " << count << " historico records in " << duration << " seconds" << std::endl;
    }
    
    void loadHistoricoTextoChunk(int min_id, int max_id) {
        std::stringstream query;
        query << "SELECT ID_CONTR, SEQ, MENSAGEM, MOTIVO, AUTORIZACAO "
              << "FROM HISTORICO_TEXTO WHERE ID_CONTR >= " << min_id 
              << " AND ID_CONTR < " << max_id
              << " ORDER BY ID_CONTR, SEQ";
        
        std::cout << "  Executing MySQL query for HISTORICO_TEXTO..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();
        
        if (mysql_query(mysql_conn, query.str().c_str())) {
            std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        std::cout << "  Fetching results..." << std::endl;
        MYSQL_RES* result = mysql_store_result(mysql_conn);
        if (!result) {
            std::cerr << "MySQL store result failed: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        MYSQL_ROW row;
        int count = 0;
        
        while ((row = mysql_fetch_row(result))) {
            std::stringstream insert;
            insert << "INSERT INTO mysql_import.historico_texto VALUES ("
                   << (row[0] ? row[0] : "0") << ", "
                   << (row[1] ? row[1] : "0") << ", "
                   << "'" << (row[2] ? escapeString(row[2]) : "") << "', "
                   << "'" << (row[3] ? escapeString(row[3]) : "") << "', "
                   << "'" << (row[4] ? escapeString(row[4]) : "") << "')";
            
            auto ch_result = executeQuery(insert.str());
            if (ch_result) {
                if (ch_result->error_message) {
                    std::cerr << "    ClickHouse insert error: " << ch_result->error_message << std::endl;
                }
                free_result_v2(ch_result);
            }
            count++;
            
            if (count % 100 == 0) {
                std::cout << "    Inserted " << count << " texto records so far..." << std::endl;
            }
        }
        
        mysql_free_result(result);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "  Loaded " << count << " historico_texto records in " << duration << " seconds" << std::endl;
    }
    
    void loadData() {
        std::cout << "\nLoading data from MySQL to ClickHouse..." << std::endl;
        
        // Get min and max ID_CONTR values
        int min_id = 0, max_id = 0;
        
        if (mysql_query(mysql_conn, "SELECT MIN(ID_CONTR), MAX(ID_CONTR) FROM HISTORICO")) {
            std::cerr << "Failed to get ID range: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        MYSQL_RES* result = mysql_store_result(mysql_conn);
        if (result) {
            MYSQL_ROW row = mysql_fetch_row(result);
            if (row && row[0] && row[1]) {
                min_id = std::stoi(row[0]);
                max_id = std::stoi(row[1]) + 1; // +1 to include the last ID
            }
            mysql_free_result(result);
        }
        
        if (min_id == 0 && max_id == 0) {
            std::cout << "No data found in HISTORICO table" << std::endl;
            return;
        }
        
        std::cout << "ID_CONTR range: [" << min_id << ", " << max_id << ")" << std::endl;
        
        // Load data in chunks
        int total_chunks = (max_id - min_id + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int current_chunk = 0;
        
        for (int current_id = min_id; current_id < max_id; current_id += CHUNK_SIZE) {
            int chunk_end = std::min(current_id + CHUNK_SIZE, max_id);
            current_chunk++;
            
            std::cout << "\nProcessing chunk " << current_chunk << "/" << total_chunks 
                      << ": ID_CONTR [" << current_id << ", " << chunk_end << ")" << std::endl;
            
            // Load HISTORICO chunk
            loadHistoricoChunk(current_id, chunk_end);
            
            // Load corresponding HISTORICO_TEXTO chunk
            loadHistoricoTextoChunk(current_id, chunk_end);
        }
        
        // Verify final counts
        auto ch_result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico");
        if (ch_result && ch_result->buf) {
            std::cout << "\nFinal historico count: " << ch_result->buf << std::endl;
            free_result_v2(ch_result);
        }
        
        ch_result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico_texto");
        if (ch_result && ch_result->buf) {
            std::cout << "Final historico_texto count: " << ch_result->buf << std::endl;
            free_result_v2(ch_result);
        }
    }
};

int main(int argc, char* argv[]) {
    std::string host = MYSQL_HOST;
    std::string user = MYSQL_USER;
    std::string password = MYSQL_PASSWORD;
    std::string database = MYSQL_DATABASE;
    bool test_mode = false;
    
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <host> <user> <password> <database> [--test]" << std::endl;
        std::cout << "\nExample: " << argv[0] << " localhost root mypassword mydatabase" << std::endl;
        std::cout << "         " << argv[0] << " localhost root mypassword mydatabase --test" << std::endl;
        std::cout << "\nUsing defaults from common.h:" << std::endl;
        std::cout << "  Host: " << host << std::endl;
        std::cout << "  User: " << user << std::endl;
        std::cout << "  Password: " << (password.empty() ? "(empty)" : "********") << std::endl;
        std::cout << "  Database: " << database << std::endl;
        
        if (argc > 1) {
            // Partial arguments provided
            std::cerr << "\nError: Please provide all 4 arguments or none" << std::endl;
            return 1;
        }
    } else {
        host = argv[1];
        user = argv[2];
        password = argv[3];
        database = argv[4];
        if (argc > 5 && std::string(argv[5]) == "--test") {
            test_mode = true;
            std::cout << "TEST MODE: Will only process 100 records per table" << std::endl;
        }
        std::cout << "Using MySQL connection: " << user << "@" << host << "/" << database << std::endl;
    }
    
    HistoricoFeeder feeder;
    
    if (test_mode) {
        feeder.setTestMode(true, 100);
    }
    
    if (!feeder.loadChdbLibrary()) {
        std::cerr << "Note: To build libchdb.so, run 'make build' in the chdb directory" << std::endl;
        return 1;
    }
    
    if (!feeder.connectToMySQL(host, user, password, database)) {
        return 1;
    }
    
    feeder.createTables();
    feeder.loadData();
    
    std::cout << "\nHistorico data feeding completed!" << std::endl;
    return 0;
}