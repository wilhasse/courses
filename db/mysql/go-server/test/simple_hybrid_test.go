package main

import (
	"fmt"
	"log"
	"os"

	"github.com/rs/zerolog"
	"mysql-server-example/pkg/hybrid"
)

func main() {
	// Simple logger
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	// Configuration
	config := hybrid.Config{
		MySQLDSN: "root:@tcp(10.1.0.7:3306)/testdb",  // Remote server with employees
		LMDBPath: "./hybrid_cache",
		Logger:   logger,
	}

	// Create handler
	handler, err := hybrid.NewHybridHandler(config)
	if err != nil {
		log.Fatal(err)
	}
	defer handler.Close()

	fmt.Println("=== Hybrid Query System Test ===\n")

	// 1. Load employees table from remote server into LMDB
	fmt.Println("1. Loading 'employees' table from 10.1.0.7 into LMDB cache...")
	err = handler.LoadTable("testdb", "employees")
	if err != nil {
		log.Fatalf("Failed to load employees table: %v", err)
	}
	fmt.Println("✓ Successfully cached employees table\n")

	// 2. Simple test - query cached employees table
	fmt.Println("2. Testing simple SELECT from cached employees:")
	query1 := "SELECT * FROM employees WHERE id <= 3"
	
	result, err := handler.ExecuteQuery(query1, "testdb")
	if err != nil {
		log.Printf("Query failed: %v", err)
	} else if result != nil {
		fmt.Printf("Query returned %d rows:\n", len(result.Rows))
		for i, row := range result.Rows {
			fmt.Printf("  Row %d: %v\n", i+1, row)
		}
	}

	// 3. Test JOIN between cached employees and local permissions
	fmt.Println("\n3. Testing JOIN query:")
	fmt.Println("   Assuming 'permissions' table exists locally with emp_id foreign key")
	
	joinQuery := `
		SELECT 
			e.name AS employee_name,
			p.permission,
			p.granted_date
		FROM employees e
		JOIN permissions p ON e.id = p.emp_id
		WHERE e.id <= 5
	`

	// Analyze the query first
	analysis, _ := handler.SQLParser.AnalyzeQuery(joinQuery, "testdb")
	fmt.Printf("\n   Query Analysis:\n")
	fmt.Printf("   - Has cached table: %v\n", analysis.HasCachedTable)
	fmt.Printf("   - Requires rewrite: %v\n", analysis.RequiresRewrite)
	
	// Execute the JOIN
	result2, err := handler.ExecuteQuery(joinQuery, "testdb")
	if err != nil {
		fmt.Printf("\n   Join query error: %v\n", err)
		fmt.Println("   (This is expected if permissions table doesn't exist locally)")
	} else if result2 != nil {
		fmt.Printf("\n   Join results (%d rows):\n", len(result2.Rows))
		for i, row := range result2.Rows {
			if i < 5 { // Show first 5 rows
				fmt.Printf("   %v\n", row)
			}
		}
	}

	fmt.Println("\n=== Test Complete ===")
}