#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <mysql/mysql.h>
#include <dlfcn.h>
#include <cstring>
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

class MinimalLoader {
private:
    MYSQL* mysql_conn;
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    MinimalLoader() : mysql_conn(nullptr), chdb_handle(nullptr) {}
    
    ~MinimalLoader() {
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
        
        // Minimal arguments - just like the working examples
        args_storage.push_back("clickhouse");
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
    
    void loadHistoricoMemory() {
        std::cout << "\nCreating HISTORICO table with Log engine..." << std::endl;
        
        auto result = executeQuery("CREATE DATABASE IF NOT EXISTS mysql_import");
        if (result) {
            free_result_v2(result);
        }
        
        // Use Log engine - Memory engine doesn't persist in chdb
        result = executeQuery(
            "CREATE TABLE IF NOT EXISTS mysql_import.historico ("
            "id_contr Int32, seq UInt16, id_funcionario Int32, "
            "id_tel Int32, data DateTime, codigo UInt16, modo String"
            ") ENGINE = Log"
        );
        
        if (result) {
            if (result->error_message) {
                std::cout << "Create table error: " << result->error_message << std::endl;
                free_result_v2(result);
                return;
            }
            std::cout << "Table created successfully" << std::endl;
            free_result_v2(result);
        }
        
        // Load first 1000 rows as a test
        std::cout << "\nLoading first 1000 rows from MySQL..." << std::endl;
        
        const char* query = "SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO "
                           "FROM HISTORICO LIMIT 1000";
        
        if (mysql_query(mysql_conn, query)) {
            std::cerr << "MySQL query failed: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        MYSQL_RES* mysql_result = mysql_store_result(mysql_conn);
        if (!mysql_result) {
            std::cerr << "MySQL store result failed: " << mysql_error(mysql_conn) << std::endl;
            return;
        }
        
        MYSQL_ROW row;
        int count = 0;
        
        while ((row = mysql_fetch_row(mysql_result))) {
            std::stringstream insert;
            insert << "INSERT INTO mysql_import.historico VALUES ("
                   << (row[0] ? row[0] : "0") << ","
                   << (row[1] ? row[1] : "0") << ","
                   << (row[2] ? row[2] : "0") << ","
                   << (row[3] ? row[3] : "0") << ","
                   << "'" << (row[4] ? row[4] : "1970-01-01 00:00:00") << "',"
                   << (row[5] ? row[5] : "0") << ","
                   << "'" << (row[6] ? row[6] : "*") << "')";
            
            auto ch_result = executeQuery(insert.str());
            if (ch_result) {
                if (ch_result->error_message) {
                    std::cout << "Row " << count << " error: " << ch_result->error_message << std::endl;
                }
                free_result_v2(ch_result);
            }
            
            count++;
            if (count % 100 == 0) {
                std::cout << "Loaded " << count << " rows..." << std::endl;
            }
        }
        
        mysql_free_result(mysql_result);
        
        std::cout << "\nTotal loaded: " << count << " rows" << std::endl;
        
        // Verify
        result = executeQuery("SELECT COUNT(*) FROM mysql_import.historico");
        if (result && result->buf) {
            std::cout << "Verified count: " << result->buf << std::endl;
            free_result_v2(result);
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <host> <user> <password> <database>" << std::endl;
        std::cout << "\nThis minimal version:" << std::endl;
        std::cout << "- Uses Memory engine instead of MergeTree" << std::endl;
        std::cout << "- No complex settings" << std::endl;
        std::cout << "- Tests basic functionality first" << std::endl;
        return 1;
    }
    
    MinimalLoader loader;
    
    if (!loader.loadChdbLibrary()) {
        return 1;
    }
    
    // Test basic functionality first
    loader.testBasicQuery();
    loader.createSimpleTable();
    
    if (!loader.connectToMySQL(argv[1], argv[2], argv[3], argv[4])) {
        return 1;
    }
    
    // Try loading with Memory engine
    loader.loadHistoricoMemory();
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}