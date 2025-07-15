package main

import (
	"fmt"
	"log"
	"os"
)

func main() {
	fmt.Println("=== Build Your Own Database - Module 01 Demo ===")
	
	// Demo 1: Basic file operations
	fmt.Println("\n1. Testing basic file operations...")
	
	testData := []byte("Hello, Database World!")
	
	// Test SaveData1 (basic write)
	err := SaveData1("test1.txt", testData)
	if err != nil {
		log.Fatalf("SaveData1 failed: %v", err)
	}
	fmt.Println("✓ SaveData1 completed")
	
	// Test SaveData2 (atomic write)
	err = SaveData2("test2.txt", testData)
	if err != nil {
		log.Fatalf("SaveData2 failed: %v", err)
	}
	fmt.Println("✓ SaveData2 completed")
	
	// Demo 2: Logging operations
	fmt.Println("\n2. Testing logging operations...")
	
	// Create log file
	logFile, err := LogCreate("demo.log")
	if err != nil {
		log.Fatalf("LogCreate failed: %v", err)
	}
	defer logFile.Close()
	
	// Append some log entries
	entries := []string{
		"Database initialization started",
		"Page size set to 4096 bytes",
		"B-tree root created",
		"Transaction #1 committed",
		"Database ready for queries",
	}
	
	for _, entry := range entries {
		err = LogAppend(logFile, entry)
		if err != nil {
			log.Fatalf("LogAppend failed: %v", err)
		}
	}
	fmt.Printf("✓ Appended %d log entries\n", len(entries))
	
	// Read back the log
	logFile.Seek(0, 0) // Reset to beginning
	readEntries, err := LogRead(logFile)
	if err != nil {
		log.Fatalf("LogRead failed: %v", err)
	}
	
	fmt.Printf("✓ Read back %d log entries:\n", len(readEntries))
	for i, entry := range readEntries {
		fmt.Printf("  [%d] %s\n", i+1, entry)
	}
	
	// Cleanup
	fmt.Println("\n3. Cleaning up test files...")
	os.Remove("test1.txt")
	os.Remove("test2.txt")
	os.Remove("demo.log")
	fmt.Println("✓ Cleanup completed")
	
	fmt.Println("\n=== Demo completed successfully! ===")
}