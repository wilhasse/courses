/**********************************************************************
 *  historico_feeder_debug.cpp  - fixed with debug logging           *
 *********************************************************************/

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
#include <memory>
#include <iomanip>
#include "common.h"

/* ------------------------------------------------------------------ */
/*                      chdb v2    public API                        */
/* ------------------------------------------------------------------ */
struct local_result_v2
{
    char      *buf;
    size_t      len;
    void       *_vec;
    double      elapsed;
    uint64_t    rows_read;
    uint64_t    bytes_read;
    char      *error_message;
};

using query_stable_v2_fn = struct local_result_v2* (*)(int, char**);
using free_result_v2_fn  = void (*)(struct local_result_v2*);

/* ------------------------------------------------------------------ */
/*                             helpers                                */
/* ------------------------------------------------------------------ */
static std::string escapeString(const std::string& src)
{
    std::string dst;
    dst.reserve(src.size());
    for (char c : src)
    {
        if (c == '\'')       dst += "\\'";
        else if (c == '\\')  dst += "\\\\";
        else                 dst += c;
    }
    return dst;
}

static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

/* ------------------------------------------------------------------ */
/*                           main class                               */
/* ------------------------------------------------------------------ */
class HistoricoFeeder
{
    /* --------------------- configuration -------------------------- */
    static constexpr int CHUNK_SIZE  = 1000;     // ID_CONTR chunk when reading MySQL
    static constexpr int BATCH_SIZE  = 1000;     // rows in every INSERT batch

    /* --------------------- data members --------------------------- */
    MYSQL                 *mysql_conn      = nullptr;

    void                  *chdb_handle     = nullptr;
    query_stable_v2_fn     query_stable_v2 = nullptr;
    free_result_v2_fn      free_result_v2  = nullptr;

    bool                   test_mode       = false;
    int                    test_limit      = 100;
    bool                   skip_texto      = false;
    long long              provided_row_count = 0;

public:
    HistoricoFeeder()  = default;
   ~HistoricoFeeder()
    {
        if (mysql_conn)  mysql_close(mysql_conn);
        if (chdb_handle) dlclose(chdb_handle);
    }

    /* -------------- command line / test helpers ------------------ */
    void setTestMode(bool mode, int limit = 100)
    {
        test_mode  = mode;
        test_limit = limit;
    }

    void setSkipTexto(bool skip) {
        skip_texto = skip;
    }

    void setRowCount(long long count) {
        provided_row_count = count;
    }

    /* -------------- chdb dynamic loading ------------------------- */
    bool loadChdbLibrary()
    {
        const char *paths[] =
        {
            "/usr/local/lib/libchdb.so",
            "./libchdb.so",
            "libchdb.so"
        };

        for (const char *p : paths)
        {
            chdb_handle = dlopen(p, RTLD_LAZY);
            if (chdb_handle)
            {
                std::cout << "[" << getCurrentTimestamp() << "] Loaded libchdb.so from \"" << p << "\"\n";
                break;
            }
        }
        if (!chdb_handle)
        {
            std::cerr << "[" << getCurrentTimestamp() << "] dlopen(libchdb.so) failed: " << dlerror() << '\n';
            return false;
        }

        query_stable_v2 = reinterpret_cast<query_stable_v2_fn>
                             (dlsym(chdb_handle, "query_stable_v2"));
        free_result_v2  = reinterpret_cast<free_result_v2_fn>
                             (dlsym(chdb_handle, "free_result_v2"));

        if (!query_stable_v2 || !free_result_v2)
        {
            std::cerr << "[" << getCurrentTimestamp() << "] dlsym() failed: " << dlerror() << '\n';
            return false;
        }
        return true;
    }

    /* ------------------ MySQL connection ------------------------- */
    bool connectToMySQL(const std::string& host,
                        const std::string& user,
                        const std::string& pass,
                        const std::string& db)
    {
        std::cout << "[" << getCurrentTimestamp() << "] Connecting to MySQL...\n";
        mysql_conn = mysql_init(nullptr);
        if (!mysql_conn)
        {
            std::cerr << "[" << getCurrentTimestamp() << "] mysql_init() failed\n";
            return false;
        }
        if (!mysql_real_connect(mysql_conn, host.c_str(), user.c_str(),
                                pass.c_str(), db.c_str(), 0, nullptr, 0))
        {
            std::cerr << "[" << getCurrentTimestamp() << "] MySQL connection failed: "
                      << mysql_error(mysql_conn) << '\n';
            return false;
        }
        std::cout << "[" << getCurrentTimestamp() << "] Connected to MySQL " << user << '@' << host
                  << '/' << db << "\n";
        return true;
    }

    /* ------------------- execute chdb query ---------------------- */
    local_result_v2* executeQuery(const std::string& query,
                                  const std::string& fmt = "CSV")
    {
        std::vector<std::string>  args_storage;
        std::vector<char*>        argv;

        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--output-format=" + fmt);
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--query=" + query);

        for (auto &s : args_storage)
            argv.push_back(s.data());

        return query_stable_v2(static_cast<int>(argv.size()), argv.data());
    }

    /* ------------------ schema management ------------------------ */
    void createTables()
    {
        std::cout << "[" << getCurrentTimestamp() << "] Creating ClickHouse tables...\n";

        auto run = [this](const std::string& q, const std::string& desc)
        {
            std::cout << "[" << getCurrentTimestamp() << "] " << desc << "...\n";
            std::unique_ptr<local_result_v2,
                            free_result_v2_fn> r(executeQuery(q), free_result_v2);
            if (r && r->error_message) {
                std::cerr << "  ClickHouse error: " << r->error_message << '\n';
            } else {
                std::cout << "  Success\n";
            }
        };

        run("CREATE DATABASE IF NOT EXISTS mysql_import", "Creating database");

        run(R"(
            CREATE TABLE IF NOT EXISTS mysql_import.historico
            (
                id_contr       Int32,
                seq            UInt16,
                id_funcionario Int32,
                id_tel         Int32,
                data           DateTime,
                codigo         UInt16,
                modo           String
            )
            ENGINE = MergeTree()
            ORDER BY (id_contr, seq)
        )", "Creating HISTORICO table");

        if (!skip_texto) {
            run(R"(
                CREATE TABLE IF NOT EXISTS mysql_import.historico_texto
                (
                    id_contr   Int32,
                    seq        UInt16,
                    mensagem   String,
                    motivo     String,
                    autorizacao String
                )
                ENGINE = MergeTree()
                ORDER BY (id_contr, seq)
            )", "Creating HISTORICO_TEXTO table");
        }

        run("TRUNCATE TABLE IF EXISTS mysql_import.historico", "Truncating HISTORICO");
        if (!skip_texto) {
            run("TRUNCATE TABLE IF EXISTS mysql_import.historico_texto", "Truncating HISTORICO_TEXTO");
        }
    }

    /* ---------------- load one HISTORICO chunk ------------------- */
    void loadHistoricoChunk(int min_id, int max_id)
    {
        std::cout << "[" << getCurrentTimestamp() << "] Loading HISTORICO chunk [" 
                  << min_id << ", " << max_id << ")...\n";

        std::ostringstream sql;
        sql << "SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO "
            << "FROM HISTORICO WHERE ID_CONTR >= " << min_id
            << " AND ID_CONTR < "  << max_id
            << " ORDER BY ID_CONTR, SEQ";

        std::cout << "[" << getCurrentTimestamp() << "] Executing MySQL query...\n";
        if (mysql_query(mysql_conn, sql.str().c_str()))
        {
            std::cerr << "[" << getCurrentTimestamp() << "] MySQL query failed: "
                      << mysql_error(mysql_conn) << '\n';
            return;
        }

        std::cout << "[" << getCurrentTimestamp() << "] Fetching results...\n";
        std::unique_ptr<MYSQL_RES, decltype(&mysql_free_result)>
            res(mysql_store_result(mysql_conn), mysql_free_result);

        if (!res)
        {
            std::cerr << "[" << getCurrentTimestamp() << "] mysql_store_result() failed: "
                      << mysql_error(mysql_conn) << '\n';
            return;
        }

        MYSQL_ROW row;
        int       total_rows  = 0;
        int       batch_rows  = 0;

        std::ostringstream batch;
        batch << "INSERT INTO mysql_import.historico VALUES ";

        auto flush_batch = [&](bool final = false)
        {
            if (batch_rows == 0) return;

            std::cout << "[" << getCurrentTimestamp() << "] Flushing batch of " 
                      << batch_rows << " rows (total: " << total_rows << ")...\n";

            std::unique_ptr<local_result_v2,
                            free_result_v2_fn> r(
                executeQuery(batch.str()), free_result_v2);

            if (r && r->error_message) {
                std::cerr << "[" << getCurrentTimestamp() << "] ClickHouse insert error: "
                          << r->error_message << '\n';
            } else {
                std::cout << "[" << getCurrentTimestamp() << "] Batch inserted successfully\n";
            }

            /* release the buffer of the used stringstream */
            std::ostringstream empty;
            batch.swap(empty);
            batch << "INSERT INTO mysql_import.historico VALUES ";
            batch_rows = 0;

            if (!final && total_rows % 10000 == 0)
                std::cout << "[" << getCurrentTimestamp() << "] Progress: inserted " 
                          << total_rows << " rows\n";
        };

        std::cout << "[" << getCurrentTimestamp() << "] Processing rows...\n";
        while ((row = mysql_fetch_row(res.get())))
        {
            if (test_mode && total_rows >= test_limit) break;

            if (batch_rows) batch << ", ";
            batch << "("
                  << (row[0] ? row[0] : "0") << ','
                  << (row[1] ? row[1] : "0") << ','
                  << (row[2] ? row[2] : "0") << ','
                  << (row[3] ? row[3] : "0") << ','
                  << '\'' << (row[4] ? row[4] : "1970-01-01 00:00:00") << '\''
                  << ','  << (row[5] ? row[5] : "0")
                  << ','  << '\'' << (row[6] ? escapeString(row[6]) : "*") << '\''
                  << ')';

            ++batch_rows;
            ++total_rows;

            if (batch_rows >= BATCH_SIZE)
                flush_batch();
        }
        flush_batch(true);   // remaining rows

        std::cout << "[" << getCurrentTimestamp() << "] HISTORICO chunk completed. Rows loaded: " 
                  << total_rows << '\n';
    }

    /* ------------ load one HISTORICO_TEXTO chunk ----------------- */
    void loadHistoricoTextoChunk(int min_id, int max_id)
    {
        if (skip_texto) {
            std::cout << "[" << getCurrentTimestamp() << "] Skipping HISTORICO_TEXTO (--skip-texto enabled)\n";
            return;
        }

        std::cout << "[" << getCurrentTimestamp() << "] Loading HISTORICO_TEXTO chunk [" 
                  << min_id << ", " << max_id << ")...\n";

        std::ostringstream sql;
        sql << "SELECT ID_CONTR, SEQ, MENSAGEM, MOTIVO, AUTORIZACAO "
            << "FROM HISTORICO_TEXTO WHERE ID_CONTR >= " << min_id
            << " AND ID_CONTR < "  << max_id
            << " ORDER BY ID_CONTR, SEQ";

        if (mysql_query(mysql_conn, sql.str().c_str()))
        {
            std::cerr << "[" << getCurrentTimestamp() << "] MySQL query failed: "
                      << mysql_error(mysql_conn) << '\n';
            return;
        }

        std::unique_ptr<MYSQL_RES, decltype(&mysql_free_result)>
            res(mysql_store_result(mysql_conn), mysql_free_result);

        if (!res)
        {
            std::cerr << "[" << getCurrentTimestamp() << "] mysql_store_result() failed: "
                      << mysql_error(mysql_conn) << '\n';
            return;
        }

        MYSQL_ROW row;
        int       total_rows  = 0;
        int       batch_rows  = 0;

        std::ostringstream batch;
        batch << "INSERT INTO mysql_import.historico_texto VALUES ";

        auto flush_batch = [&](bool final = false)
        {
            if (batch_rows == 0) return;

            std::cout << "[" << getCurrentTimestamp() << "] Flushing TEXTO batch of " 
                      << batch_rows << " rows...\n";

            std::unique_ptr<local_result_v2,
                            free_result_v2_fn> r(
                executeQuery(batch.str()), free_result_v2);

            if (r && r->error_message)
                std::cerr << "[" << getCurrentTimestamp() << "] ClickHouse insert error: "
                          << r->error_message << '\n';

            /* release the buffer */
            std::ostringstream empty;
            batch.swap(empty);
            batch << "INSERT INTO mysql_import.historico_texto VALUES ";
            batch_rows = 0;

            if (!final && total_rows % 10000 == 0)
                std::cout << "[" << getCurrentTimestamp() << "] TEXTO progress: " 
                          << total_rows << " rows\n";
        };

        while ((row = mysql_fetch_row(res.get())))
        {
            if (test_mode && total_rows >= test_limit) break;

            if (batch_rows) batch << ", ";
            batch << "("
                  << (row[0] ? row[0] : "0") << ','
                  << (row[1] ? row[1] : "0") << ','
                  << '\'' << (row[2] ? escapeString(row[2]) : "") << '\''
                  << ','  << '\'' << (row[3] ? escapeString(row[3]) : "") << '\''
                  << ','  << '\'' << (row[4] ? escapeString(row[4]) : "") << '\''
                  << ')';

            ++batch_rows;
            ++total_rows;

            if (batch_rows >= 100)  // Smaller batch size for text data
                flush_batch();
        }
        flush_batch(true);

        std::cout << "[" << getCurrentTimestamp() << "] HISTORICO_TEXTO chunk completed. Rows loaded: " 
                  << total_rows << '\n';
    }

    /* ----------------- high-level import loop -------------------- */
    void loadData()
    {
        /* obtain id range ------------------------------------------------ */
        int min_id = 0, max_id = 0;
        
        std::cout << "[" << getCurrentTimestamp() << "] Getting ID range from HISTORICO...\n";
        if (mysql_query(mysql_conn,
                        "SELECT MIN(ID_CONTR), MAX(ID_CONTR) FROM HISTORICO"))
        {
            std::cerr << "[" << getCurrentTimestamp() << "] Failed to read ID range: "
                      << mysql_error(mysql_conn) << '\n';
            return;
        }
        {
            std::unique_ptr<MYSQL_RES, decltype(&mysql_free_result)>
                res(mysql_store_result(mysql_conn), mysql_free_result);

            if (res)
            {
                if (MYSQL_ROW r = mysql_fetch_row(res.get()))
                {
                    if (r[0]) min_id = std::stoi(r[0]);
                    if (r[1]) max_id = std::stoi(r[1]) + 1;
                }
            }
        }
        if (min_id >= max_id)
        {
            std::cout << "[" << getCurrentTimestamp() << "] No data in HISTORICO table\n";
            return;
        }

        std::cout << "[" << getCurrentTimestamp() << "] Importing ID_CONTR range [" << min_id
                  << ", " << max_id << ")\n";

        if (provided_row_count > 0) {
            std::cout << "[" << getCurrentTimestamp() << "] Using provided row count: " 
                      << provided_row_count << "\n";
        }

        /* iterate through chunks ---------------------------------------- */
        int chunk_num = 0;
        int total_chunks = (max_id - min_id + CHUNK_SIZE - 1) / CHUNK_SIZE;
        
        for (int start = min_id; start < max_id; start += CHUNK_SIZE)
        {
            chunk_num++;
            int end = std::min(start + CHUNK_SIZE, max_id);
            std::cout << "\n[" << getCurrentTimestamp() << "] Processing chunk " 
                      << chunk_num << "/" << total_chunks 
                      << " ID_CONTR [" << start << ", " << end << ")\n";

            loadHistoricoChunk(start, end);
            loadHistoricoTextoChunk(start, end);
        }

        /* verify counts -------------------------------------------------- */
        std::cout << "\n[" << getCurrentTimestamp() << "] Verifying final counts...\n";
        auto chk = [this](const std::string& table)
        {
            std::unique_ptr<local_result_v2, free_result_v2_fn> r(
                executeQuery("SELECT COUNT(*) FROM " + table), free_result_v2);
            if (r && r->buf)
                std::cout << "[" << getCurrentTimestamp() << "] Rows in " << table 
                          << ": " << r->buf;
        };
        chk("mysql_import.historico");
        if (!skip_texto) {
            chk("mysql_import.historico_texto");
        }
    }
};

/* ================================================================== */
/*                               main                                 */
/* ================================================================== */
int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <host> <user> <password> <database> [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --skip-texto           Skip loading HISTORICO_TEXTO table\n";
        std::cout << "  --row-count <count>    Provide row count for progress tracking\n";
        std::cout << "  --test                 Test mode (limit to 100 rows)\n";
        return 1;
    }

    std::string host     = argv[1];
    std::string user     = argv[2];
    std::string password = argv[3];
    std::string database = argv[4];

    HistoricoFeeder feeder;

    // Parse options
    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--skip-texto") {
            feeder.setSkipTexto(true);
            std::cout << "[" << getCurrentTimestamp() << "] --skip-texto enabled\n";
        } else if (arg == "--row-count" && i + 1 < argc) {
            feeder.setRowCount(std::stoll(argv[++i]));
        } else if (arg == "--test") {
            feeder.setTestMode(true);
            std::cout << "[" << getCurrentTimestamp() << "] Test mode enabled\n";
        }
    }

    std::cout << "\n[" << getCurrentTimestamp() << "] Starting historico_feeder_debug...\n";

    if (!feeder.loadChdbLibrary()) return 1;
    if (!feeder.connectToMySQL(host, user, password, database)) return 1;

    feeder.createTables();
    feeder.loadData();

    std::cout << "\n[" << getCurrentTimestamp() << "] Historico feeding completed.\n";
    return 0;
}