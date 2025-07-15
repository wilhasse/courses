#include <iostream>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sstream>
#include <chrono>
#include <iomanip>
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

class SqlExecutor {
private:
    void* chdb_handle;
    query_stable_v2_fn query_stable_v2;
    free_result_v2_fn free_result_v2;
    
public:
    SqlExecutor() : chdb_handle(nullptr) {}
    
    ~SqlExecutor() {
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
                break;
            }
        }
        
        if (!chdb_handle) {
            std::cerr << "Error: Failed to load libchdb.so: " << dlerror() << std::endl;
            return false;
        }
        
        query_stable_v2 = (query_stable_v2_fn)dlsym(chdb_handle, "query_stable_v2");
        free_result_v2 = (free_result_v2_fn)dlsym(chdb_handle, "free_result_v2");
        
        if (!query_stable_v2 || !free_result_v2) {
            std::cerr << "Error: Failed to load functions: " << dlerror() << std::endl;
            return false;
        }
        
        return true;
    }
    
    struct local_result_v2* executeQuery(const std::string& query, const std::string& format = "Pretty") {
        std::vector<char*> argv;
        std::vector<std::string> args_storage;
        
        args_storage.push_back("clickhouse");
        args_storage.push_back("--multiquery");
        args_storage.push_back("--path=" + CHDB_PATH);
        args_storage.push_back("--output-format=" + format);
        args_storage.push_back("--query=" + query);
        
        for (auto& arg : args_storage) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        return query_stable_v2(argv.size(), argv.data());
    }
    
    void runQuery(const std::string& query, const std::string& format) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto result = executeQuery(query, format);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (result) {
            if (result->error_message) {
                std::cerr << "\nError: " << result->error_message << std::endl;
            } else if (result->buf && result->len > 0) {
                std::cout << result->buf;
                if (result->buf[result->len - 1] != '\n') {
                    std::cout << std::endl;
                }
            } else {
                std::cout << "(empty result)" << std::endl;
            }
            
            // Show statistics in verbose mode
            if (format == "Pretty") {
                std::cout << "\n";
                std::cout << "Elapsed: " << std::fixed << std::setprecision(3) 
                          << (duration / 1000.0) << " sec. ";
                if (result->rows_read > 0) {
                    std::cout << "Processed " << result->rows_read << " rows, ";
                    
                    // Format bytes
                    double bytes = result->bytes_read;
                    if (bytes > 1024 * 1024 * 1024) {
                        std::cout << std::fixed << std::setprecision(2) 
                                  << (bytes / (1024 * 1024 * 1024)) << " GB";
                    } else if (bytes > 1024 * 1024) {
                        std::cout << std::fixed << std::setprecision(2) 
                                  << (bytes / (1024 * 1024)) << " MB";
                    } else if (bytes > 1024) {
                        std::cout << std::fixed << std::setprecision(2) 
                                  << (bytes / 1024) << " KB";
                    } else {
                        std::cout << (int)bytes << " B";
                    }
                }
                std::cout << std::endl;
            }
            
            free_result_v2(result);
        } else {
            std::cerr << "Error: Query execution failed" << std::endl;
        }
    }
    
    void showHelp(const std::string& program) {
        std::cout << "Usage: " << program << " [options] <query>" << std::endl;
        std::cout << "\nOptions:" << std::endl;
        std::cout << "  -f, --format <format>  Output format (default: Pretty)" << std::endl;
        std::cout << "  -h, --help            Show this help message" << std::endl;
        std::cout << "\nAvailable formats:" << std::endl;
        std::cout << "  Pretty              Human-readable table format (default)" << std::endl;
        std::cout << "  TabSeparated, TSV   Tab-separated values" << std::endl;
        std::cout << "  CSV                 Comma-separated values" << std::endl;
        std::cout << "  JSON                JSON format" << std::endl;
        std::cout << "  JSONCompact         Compact JSON format" << std::endl;
        std::cout << "  Values              Insert-ready values" << std::endl;
        std::cout << "  Vertical            One value per line" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << program << " \"SELECT COUNT(*) FROM mysql_import.historico\"" << std::endl;
        std::cout << "  " << program << " -f TSV \"SELECT * FROM mysql_import.historico LIMIT 10\"" << std::endl;
        std::cout << "  " << program << " --format JSON \"SELECT codigo, COUNT(*) as cnt FROM mysql_import.historico GROUP BY codigo ORDER BY cnt DESC LIMIT 5\"" << std::endl;
        std::cout << "\nTips:" << std::endl;
        std::cout << "  - Use quotes around your SQL query" << std::endl;
        std::cout << "  - Use semicolon for multiple queries" << std::endl;
        std::cout << "  - Pipe output for further processing: " << program << " -f TSV \"SELECT ...\" | awk ..." << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        SqlExecutor executor;
        executor.showHelp(argv[0]);
        return 1;
    }
    
    std::string query;
    std::string format = "Pretty";
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            SqlExecutor executor;
            executor.showHelp(argv[0]);
            return 0;
        } else if ((arg == "-f" || arg == "--format") && i + 1 < argc) {
            format = argv[++i];
        } else if (arg[0] != '-') {
            // This is the query
            query = arg;
            // Concatenate remaining arguments as part of the query
            for (int j = i + 1; j < argc; j++) {
                query += " " + std::string(argv[j]);
            }
            break;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }
    
    if (query.empty()) {
        std::cerr << "Error: No query provided" << std::endl;
        std::cerr << "Use -h or --help for usage information" << std::endl;
        return 1;
    }
    
    SqlExecutor executor;
    
    if (!executor.loadChdbLibrary()) {
        return 1;
    }
    
    executor.runQuery(query, format);
    
    return 0;
}