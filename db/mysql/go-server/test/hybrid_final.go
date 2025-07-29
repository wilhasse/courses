package main

import (
	"fmt"
	"log"
	"os"

	_ "github.com/go-sql-driver/mysql"
	"github.com/rs/zerolog"
	"mysql-server-example/pkg/hybrid"
)

func main() {
	// Setup logger
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	// Remote MySQL server  
	remoteDSN := "root:@tcp(10.1.0.7:3306)/testdb"
	
	fmt.Println("=== Hybrid Query System Test ===")
	fmt.Println("Remote MySQL: 10.1.0.7")
	fmt.Println("Cached Table: employees (from remote)")
	fmt.Println("Remote Table: employee_notes (stays on remote)")
	fmt.Println()

	// Create hybrid handler
	config := hybrid.Config{
		MySQLDSN: remoteDSN,
		LMDBPath: "./final_test_cache",
		Logger:   logger,
	}
	defer os.RemoveAll("./final_test_cache")

	handler, err := hybrid.NewHybridHandler(config)
	if err != nil {
		log.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer handler.Close()

	// Step 1: Load employees table into LMDB cache
	fmt.Println("1. Loading 'employees' table from remote MySQL into LMDB cache...")
	err = handler.LoadTable("testdb", "employees")
	if err != nil {
		log.Fatalf("Failed to load employees table: %v", err)
	}
	fmt.Println("✓ Successfully cached employees table")
	fmt.Println()

	// Step 2: Test simple query on cached data
	fmt.Println("2. Testing SELECT from cached employees table:")
	query1 := "SELECT id, first_name, last_name, department FROM employees WHERE id <= 5"
	
	result, err := handler.ExecuteQuery(query1, "testdb")
	if err != nil {
		log.Printf("Query failed: %v", err)
	} else if result != nil {
		fmt.Printf("Query returned %d rows:\n", len(result.Rows))
		fmt.Printf("Columns: %v\n", result.Columns)
		for _, row := range result.Rows {
			// Convert byte arrays to strings for readable output
			id := row[0]
			firstName := string(row[1].([]byte))
			lastName := string(row[2].([]byte))
			department := string(row[3].([]byte))
			fmt.Printf("  ID: %v, Name: %s %s, Dept: %s\n", id, firstName, lastName, department)
		}
	}
	fmt.Println()

	// Step 3: Test JOIN between cached employees and remote employee_notes
	fmt.Println("3. Testing JOIN between cached employees and remote employee_notes:")
	
	// Register employees as cached
	handler.SQLParser.RegisterCachedTable("testdb", "employees")
	
	joinQuery := `
		SELECT 
			e.id,
			e.first_name,
			e.last_name,
			n.note,
			n.created_at
		FROM employees e
		JOIN employee_notes n ON e.id = n.emp_id
		WHERE e.id <= 5
		ORDER BY e.id, n.created_at
		LIMIT 10
	`

	// Analyze the query
	fmt.Println("\nQuery Analysis:")
	analysis, err := handler.SQLParser.AnalyzeQuery(joinQuery, "testdb")
	if err != nil {
		log.Printf("Failed to analyze query: %v", err)
	} else {
		fmt.Printf("- Has cached table: %v\n", analysis.HasCachedTable)
		fmt.Printf("- Cached tables: %v\n", analysis.CachedTables)
		fmt.Printf("- Remote tables: %v\n", analysis.RemoteTables)
		fmt.Printf("- Is join query: %v\n", analysis.IsJoinQuery)
		fmt.Printf("- Requires rewrite: %v\n", analysis.RequiresRewrite)
	}

	// Show what the rewriter does
	fmt.Println("\nQuery Rewriting:")
	rewriteResult, err := handler.QueryRewriter.RewriteQuery(joinQuery, "testdb")
	if err != nil {
		log.Printf("Failed to rewrite query: %v", err)
	} else {
		fmt.Println("Original tables: employees (cached), employee_notes (remote)")
		fmt.Printf("Rewritten query for MySQL: %s\n", rewriteResult.RemoteQuery)
		fmt.Printf("Cached tables to query: %v\n", rewriteResult.CachedTableNames)
	}

	// Execute the hybrid JOIN
	fmt.Println("\nExecuting Hybrid JOIN:")
	result2, err := handler.ExecuteQuery(joinQuery, "testdb")
	if err != nil {
		fmt.Printf("ERROR: Join query failed: %v\n", err)
		fmt.Println("\nMake sure employee_notes table exists on remote server.")
		fmt.Println("Run: mysql -h 10.1.0.7 -u root testdb < test/create_remote_test_tables.sql")
	} else if result2 != nil {
		fmt.Printf("\n✓ JOIN successful! Results:\n")
		fmt.Printf("Columns: %v\n", result2.Columns)
		fmt.Println("\nData:")
		for i, row := range result2.Rows {
			// Convert byte arrays to strings
			id := row[0]
			firstName := string(row[1].([]byte))
			lastName := string(row[2].([]byte))
			note := string(row[3].([]byte))
			createdAt := row[4]
			
			fmt.Printf("Employee #%v (%s %s): %s [Created: %v]\n", 
				id, firstName, lastName, note, createdAt)
			
			if i >= 9 { // Show first 10 rows
				if len(result2.Rows) > 10 {
					fmt.Printf("... (%d more rows)\n", len(result2.Rows)-10)
				}
				break
			}
		}
		fmt.Printf("\nTotal rows returned: %d\n", len(result2.Rows))
	}

	// Performance comparison
	fmt.Println("\n4. Performance Notes:")
	fmt.Println("- Employees table is cached locally in LMDB (fast access)")
	fmt.Println("- Employee_notes remains on remote MySQL")
	fmt.Println("- JOIN is performed in-memory after fetching both datasets")
	fmt.Println("- This approach is beneficial when:")
	fmt.Println("  * employees table is frequently accessed")
	fmt.Println("  * employees table is relatively static")
	fmt.Println("  * Network latency to remote MySQL is high")

	fmt.Println("\n=== Test Complete ===")
}