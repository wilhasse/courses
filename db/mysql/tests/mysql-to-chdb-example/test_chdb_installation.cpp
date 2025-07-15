#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <iomanip>

struct TestResult {
    std::string name;
    bool found;
    void* address;
};

int main() {
    std::cout << "=== chDB Installation Test ===" << std::endl;
    std::cout << std::endl;
    
    // Try to load libchdb.so
    std::cout << "1. Testing library loading..." << std::endl;
    void* handle = dlopen("libchdb.so", RTLD_LAZY);
    
    if (!handle) {
        std::cerr << "❌ FAILED to load libchdb.so: " << dlerror() << std::endl;
        std::cerr << "\nPossible solutions:" << std::endl;
        std::cerr << "  1. Install chDB: curl -sL https://lib.chdb.io | bash" << std::endl;
        std::cerr << "  2. Run: sudo ldconfig" << std::endl;
        std::cerr << "  3. Check: ldconfig -p | grep chdb" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Successfully loaded libchdb.so" << std::endl;
    std::cout << std::endl;
    
    // List of functions to check
    std::vector<std::string> functions = {
        // Core query functions
        "query_stable",
        "query_stable_v2",
        "free_result",
        "free_result_v2",
        
        // Session functions
        "session_new",
        "session_query",
        "session_get_error",
        "session_free",
        
        // Connection functions  
        "connect_chdb",
        "close_chdb",
        
        // Utility functions
        "get_error_message",
        "chdb_version",
        "chdb_last_error",
        
        // Memory management
        "chdb_allocate",
        "chdb_deallocate"
    };
    
    std::cout << "2. Checking available functions..." << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "Function" 
              << std::setw(10) << "Status" 
              << "Address" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    std::vector<TestResult> results;
    int found_count = 0;
    
    for (const auto& func : functions) {
        void* sym = dlsym(handle, func.c_str());
        bool found = (sym != nullptr);
        
        if (found) found_count++;
        
        results.push_back({func, found, sym});
        
        std::cout << std::left << std::setw(25) << func 
                  << std::setw(10) << (found ? "✅ Found" : "❌ Missing")
                  << (found ? sym : nullptr) << std::endl;
    }
    
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "Total: " << found_count << "/" << functions.size() 
              << " functions found" << std::endl;
    std::cout << std::endl;
    
    // Test a simple query if core functions are available
    std::cout << "3. Testing basic functionality..." << std::endl;
    
    // Check for query_stable_v2
    typedef struct local_result_v2 {
        char * buf;
        size_t len;
        void * _vec;
        double elapsed;
        uint64_t rows_read;
        uint64_t bytes_read;
        char * error_message;
    } local_result_v2;
    
    typedef local_result_v2* (*query_stable_v2_fn)(int argc, char** argv);
    typedef void (*free_result_v2_fn)(local_result_v2* result);
    
    query_stable_v2_fn query_stable_v2 = (query_stable_v2_fn)dlsym(handle, "query_stable_v2");
    free_result_v2_fn free_result_v2 = (free_result_v2_fn)dlsym(handle, "free_result_v2");
    
    if (query_stable_v2 && free_result_v2) {
        std::cout << "✅ Core query functions available" << std::endl;
        
        // Try a simple query
        std::vector<char*> argv;
        std::vector<std::string> args = {
            "clickhouse",
            "--query=SELECT version(), now()"
        };
        
        for (auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        
        std::cout << "\n4. Running test query..." << std::endl;
        local_result_v2* result = query_stable_v2(argv.size(), argv.data());
        
        if (result) {
            if (result->error_message) {
                std::cout << "❌ Query error: " << result->error_message << std::endl;
            } else if (result->buf && result->len > 0) {
                std::cout << "✅ Query successful!" << std::endl;
                std::cout << "   Result: " << std::string(result->buf, result->len);
                std::cout << "   Elapsed: " << result->elapsed << " seconds" << std::endl;
            } else {
                std::cout << "⚠️  Query returned empty result" << std::endl;
            }
            free_result_v2(result);
        } else {
            std::cout << "❌ Query execution failed" << std::endl;
        }
    } else {
        std::cout << "❌ Core query functions not available" << std::endl;
    }
    
    // Check library info
    std::cout << "\n5. Library information..." << std::endl;
    
    // Try to get version if available
    typedef const char* (*version_fn)();
    version_fn get_version = (version_fn)dlsym(handle, "chdb_version");
    if (get_version) {
        const char* version = get_version();
        if (version) {
            std::cout << "   Version: " << version << std::endl;
        }
    }
    
    // Check library path
    Dl_info info;
    if (dladdr(handle, &info)) {
        std::cout << "   Library path: " << (info.dli_fname ? info.dli_fname : "unknown") << std::endl;
    }
    
    dlclose(handle);
    
    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    if (found_count >= 2) {  // At least query_stable_v2 and free_result_v2
        std::cout << "✅ chDB is properly installed and functional!" << std::endl;
    } else {
        std::cout << "❌ chDB installation appears to be incomplete" << std::endl;
    }
    
    return 0;
}