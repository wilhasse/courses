#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <mutex>
#include <iostream>

class Logger {
private:
    std::ofstream logFile;
    std::mutex logMutex;
    bool consoleOutput;
    std::string logFilePath;

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string formatSize(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unitIndex = 0;
        double size = static_cast<double>(bytes);
        
        while (size >= 1024.0 && unitIndex < 3) {
            size /= 1024.0;
            unitIndex++;
        }
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
        return ss.str();
    }

public:
    Logger(const std::string& logPath = "chdb_queries.log", bool enableConsole = true) 
        : logFilePath(logPath), consoleOutput(enableConsole) {
        logFile.open(logPath, std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Warning: Could not open log file: " << logPath << std::endl;
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    void logQuery(const std::string& query, double elapsedMs, 
                  size_t resultRows = 0, size_t resultBytes = 0,
                  const std::string& format = "JSON") {
        std::lock_guard<std::mutex> lock(logMutex);
        
        std::stringstream logEntry;
        logEntry << "[" << getCurrentTimestamp() << "] "
                 << "QUERY | "
                 << "Time: " << std::fixed << std::setprecision(3) << elapsedMs << "ms | "
                 << "Rows: " << resultRows << " | "
                 << "Size: " << formatSize(resultBytes) << " | "
                 << "Format: " << format << " | "
                 << "Query: " << query;
        
        std::string entry = logEntry.str();
        
        if (consoleOutput) {
            std::cout << entry << std::endl;
        }
        
        if (logFile.is_open()) {
            logFile << entry << std::endl;
            logFile.flush();
        }
    }

    void logInfo(const std::string& message) {
        std::lock_guard<std::mutex> lock(logMutex);
        
        std::stringstream logEntry;
        logEntry << "[" << getCurrentTimestamp() << "] INFO | " << message;
        
        std::string entry = logEntry.str();
        
        if (consoleOutput) {
            std::cout << entry << std::endl;
        }
        
        if (logFile.is_open()) {
            logFile << entry << std::endl;
            logFile.flush();
        }
    }

    void logError(const std::string& message) {
        std::lock_guard<std::mutex> lock(logMutex);
        
        std::stringstream logEntry;
        logEntry << "[" << getCurrentTimestamp() << "] ERROR | " << message;
        
        std::string entry = logEntry.str();
        
        if (consoleOutput) {
            std::cerr << entry << std::endl;
        }
        
        if (logFile.is_open()) {
            logFile << entry << std::endl;
            logFile.flush();
        }
    }
};

#endif // LOGGER_H