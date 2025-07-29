package main

import (
	"database/sql"
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

	// Remote MySQL server with employees table
	remoteDSN := "root:@tcp(10.1.0.7:3306)/testdb"
	
	// Local LMDB cache path
	lmdbPath := "./test_hybrid_cache"
	defer os.RemoveAll(lmdbPath) // Clean up after test

	// Create hybrid handler
	config := hybrid.Config{
		MySQLDSN: remoteDSN,
		LMDBPath: lmdbPath,
		Logger:   logger,
	}

	handler, err := hybrid.NewHybridHandler(config)
	if err != nil {
		log.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer handler.Close()

	// Step 1: Load employees table from remote server into LMDB cache
	fmt.Println("=== Loading employees table from remote server into LMDB ===")
	err = handler.LoadTable("testdb", "employees")
	if err != nil {
		log.Fatalf("Failed to load employees table: %v", err)
	}
	fmt.Println("✓ Employees table loaded into LMDB cache")

	// Step 2: Create local permissions table (in your local MySQL)
	fmt.Println("\n=== Creating local permissions table ===")
	localDSN := "root:@tcp(localhost:3306)/testdb"
	localDB, err := sql.Open("mysql", localDSN)
	if err != nil {
		log.Fatalf("Failed to connect to local MySQL: %v", err)
	}
	defer localDB.Close()

	// Create permissions table
	_, err = localDB.Exec(`
		CREATE TABLE IF NOT EXISTS permissions (
			id INT PRIMARY KEY AUTO_INCREMENT,
			emp_id INT NOT NULL,
			permission VARCHAR(50) NOT NULL,
			granted_date DATE
		)
	`)
	if err != nil {
		log.Printf("Warning: Failed to create permissions table: %v", err)
	}

	// Insert sample permissions data
	_, err = localDB.Exec(`
		INSERT IGNORE INTO permissions (emp_id, permission, granted_date) VALUES
		(1, 'READ', '2024-01-01'),
		(1, 'WRITE', '2024-01-15'),
		(2, 'READ', '2024-02-01'),
		(3, 'ADMIN', '2024-03-01'),
		(4, 'READ', '2024-01-10'),
		(5, 'WRITE', '2024-02-20')
	`)
	if err != nil {
		log.Printf("Warning: Failed to insert permissions data: %v", err)
	}
	fmt.Println("✓ Permissions table created locally")

	// Step 3: Test simple query on cached employees table
	fmt.Println("\n=== Testing simple query on cached employees table ===")
	query1 := "SELECT id, name FROM employees LIMIT 5"
	result1, err := handler.ExecuteQuery(query1, "testdb")
	if err != nil {
		log.Printf("Simple query failed: %v", err)
	} else if result1 != nil {
		fmt.Printf("Query: %s\n", query1)
		fmt.Printf("Columns: %v\n", result1.Columns)
		for i, row := range result1.Rows {
			if i < 5 {
				fmt.Printf("Row %d: %v\n", i+1, row)
			}
		}
	}

	// Step 4: Test JOIN query between cached employees and local permissions
	fmt.Println("\n=== Testing JOIN between cached employees and local permissions ===")
	
	// Register employees as cached in the parser
	handler.SQLParser.RegisterCachedTable("testdb", "employees")
	
	joinQuery := `
		SELECT 
			e.id,
			e.name,
			p.permission,
			p.granted_date
		FROM employees e
		JOIN permissions p ON e.id = p.emp_id
		ORDER BY e.id, p.granted_date
	`

	// First, let's analyze the query
	analysis, err := handler.SQLParser.AnalyzeQuery(joinQuery, "testdb")
	if err != nil {
		log.Printf("Failed to analyze query: %v", err)
	} else {
		fmt.Println("\nQuery Analysis:")
		fmt.Printf("- Has cached table: %v\n", analysis.HasCachedTable)
		fmt.Printf("- Cached tables: %v\n", analysis.CachedTables)
		fmt.Printf("- Remote tables: %v\n", analysis.RemoteTables)
		fmt.Printf("- Is join query: %v\n", analysis.IsJoinQuery)
		fmt.Printf("- Requires rewrite: %v\n", analysis.RequiresRewrite)
	}

	// Execute the hybrid join query
	result2, err := handler.ExecuteQuery(joinQuery, "testdb")
	if err != nil {
		log.Printf("Join query failed: %v", err)
	} else if result2 != nil {
		fmt.Printf("\nJoin Query Results:\n")
		fmt.Printf("Columns: %v\n", result2.Columns)
		fmt.Println("Data:")
		for i, row := range result2.Rows {
			fmt.Printf("Employee ID: %v, Name: %v, Permission: %v, Granted: %v\n",
				row[0], row[1], row[2], row[3])
			if i >= 9 { // Show first 10 rows
				fmt.Printf("... (%d more rows)\n", len(result2.Rows)-10)
				break
			}
		}
	}

	// Step 5: Show what the rewriter does
	fmt.Println("\n=== Query Rewriting Demo ===")
	rewriteResult, err := handler.QueryRewriter.RewriteQuery(joinQuery, "testdb")
	if err != nil {
		log.Printf("Failed to rewrite query: %v", err)
	} else {
		fmt.Println("Original Query:")
		fmt.Println(joinQuery)
		fmt.Println("\nRewritten Query (for MySQL without cached tables):")
		fmt.Println(rewriteResult.RemoteQuery)
		fmt.Println("\nCached Tables to Query from LMDB:")
		fmt.Println(rewriteResult.CachedTableNames)
	}

	// Step 6: Performance comparison
	fmt.Println("\n=== Performance Comparison ===")
	
	// Direct query to remote server
	remoteDB, _ := sql.Open("mysql", remoteDSN)
	defer remoteDB.Close()
	
	var count int
	err = remoteDB.QueryRow("SELECT COUNT(*) FROM employees").Scan(&count)
	if err == nil {
		fmt.Printf("Total employees in remote database: %d\n", count)
	}

	// Stats
	stats := handler.GetStats()
	fmt.Printf("\nHybrid Cache Stats:\n")
	fmt.Printf("- Enabled: %v\n", stats.Enabled)
	fmt.Printf("- Cached tables: %v\n", stats.CachedTables)
}