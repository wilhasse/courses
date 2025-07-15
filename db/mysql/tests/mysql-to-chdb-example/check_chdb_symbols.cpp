#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <string>

int main() {
    void* handle = dlopen("libchdb.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }
    
    std::cout << "Library loaded successfully!" << std::endl;
    
    // List of function names to check
    std::vector<std::string> functions = {
        "Execute",
        "Query", 
        "QuerySession",
        "FreeResult",
        "query_stable",
        "query_stable_v2",
        "free_result_v2",
        "chdb_query",
        "chdb_free_result"
    };
    
    std::cout << "\nChecking available functions:" << std::endl;
    for (const auto& func : functions) {
        void* sym = dlsym(handle, func.c_str());
        if (sym) {
            std::cout << "✓ " << func << " - Found at " << sym << std::endl;
        } else {
            std::cout << "✗ " << func << " - Not found" << std::endl;
        }
    }
    
    dlclose(handle);
    return 0;
}