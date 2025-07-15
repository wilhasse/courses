#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <cstring>

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

int main() {
    void* handle = dlopen("libchdb.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }
    
    auto query_stable_v2 = (query_stable_v2_fn)dlsym(handle, "query_stable_v2");
    auto free_result_v2 = (free_result_v2_fn)dlsym(handle, "free_result_v2");
    
    if (!query_stable_v2 || !free_result_v2) {
        std::cerr << "Failed to load functions: " << dlerror() << std::endl;
        return 1;
    }
    
    // Check row counts
    std::vector<std::string> queries = {
        "SELECT COUNT(*) as count FROM mysql_import.historico",
        "SELECT MIN(id_contr) as min_id, MAX(id_contr) as max_id FROM mysql_import.historico",
        "SELECT COUNT(*) as count FROM mysql_import.historico_texto"
    };
    
    for (const auto& query : queries) {
        std::vector<char*> argv;
        std::vector<std::string> args = {
            "clickhouse",
            "--path=./clickhouse_data",
            "--query=" + query
        };
        
        for (auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "\nQuery: " << query << std::endl;
        auto result = query_stable_v2(argv.size(), argv.data());
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
    
    dlclose(handle);
    return 0;
}