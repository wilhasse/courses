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

	// Remote MySQL server
	remoteDSN := "root:@tcp(10.1.0.7:3306)/testdb"
	
	// First, check what columns the employees table has
	fmt.Println("=== Checking remote employees table structure ===")
	remoteDB, err := sql.Open("mysql", remoteDSN)
	if err != nil {
		log.Fatal(err)
	}
	defer remoteDB.Close()

	// Get column information
	rows, err := remoteDB.Query(`
		SELECT COLUMN_NAME, DATA_TYPE 
		FROM INFORMATION_SCHEMA.COLUMNS 
		WHERE TABLE_SCHEMA = 'testdb' AND TABLE_NAME = 'employees'
		ORDER BY ORDINAL_POSITION
	`)
	if err != nil {
		log.Fatalf("Failed to get table info: %v", err)
	}

	var columns []string
	fmt.Println("Employees table columns:")
	for rows.Next() {
		var colName, dataType string
		rows.Scan(&colName, &dataType)
		fmt.Printf("  - %s (%s)\n", colName, dataType)
		columns = append(columns, colName)
	}
	rows.Close()

	// Get sample data
	fmt.Println("\nSample data from employees:")
	rows, err = remoteDB.Query("SELECT * FROM employees LIMIT 3")
	if err != nil {
		log.Fatal(err)
	}
	
	// Get column names from result set
	cols, _ := rows.Columns()
	fmt.Printf("Columns: %v\n", cols)
	
	// Scan and display rows
	for rows.Next() {
		// Create a slice of interface{} to hold the values
		values := make([]interface{}, len(cols))
		valuePtrs := make([]interface{}, len(cols))
		for i := range values {
			valuePtrs[i] = &values[i]
		}
		
		rows.Scan(valuePtrs...)
		fmt.Printf("Row: %v\n", values)
	}
	rows.Close()

	// Now proceed with hybrid test
	fmt.Println("\n=== Setting up Hybrid Query System ===")
	
	// Create hybrid handler
	config := hybrid.Config{
		MySQLDSN: remoteDSN,
		LMDBPath: "./adaptive_test_cache",
		Logger:   logger,
	}
	defer os.RemoveAll("./adaptive_test_cache")

	handler, err := hybrid.NewHybridHandler(config)
	if err != nil {
		log.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer handler.Close()

	// Load employees table
	fmt.Println("\nLoading employees table into LMDB cache...")
	err = handler.LoadTable("testdb", "employees")
	if err != nil {
		log.Fatalf("Failed to load employees table: %v", err)
	}
	fmt.Println("✓ Successfully loaded employees table")

	// Test simple query on cached data
	fmt.Println("\n=== Testing cached data access ===")
	testQuery := "SELECT * FROM employees LIMIT 5"
	result, err := handler.ExecuteQuery(testQuery, "testdb")
	if err != nil {
		log.Printf("Query failed: %v", err)
	} else if result != nil {
		fmt.Printf("Query returned %d rows with columns: %v\n", len(result.Rows), result.Columns)
		for i, row := range result.Rows {
			if i < 3 {
				fmt.Printf("Row %d: %v\n", i+1, row)
			}
		}
	}

	// Create a local table that references employees
	fmt.Println("\n=== Creating local table with employee references ===")
	localDB, err := sql.Open("mysql", "root:@tcp(localhost:3306)/testdb")
	if err != nil {
		log.Fatal(err)
	}
	defer localDB.Close()

	// Find the first column (likely to be the ID/primary key)
	idColumn := "id"
	if len(columns) > 0 {
		idColumn = columns[0]
		fmt.Printf("Using '%s' as the ID column\n", idColumn)
	}

	// Create a simple employee_notes table
	_, err = localDB.Exec(fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS employee_notes (
			note_id INT PRIMARY KEY AUTO_INCREMENT,
			emp_id INT NOT NULL,
			note TEXT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)
	`))
	if err != nil {
		log.Printf("Warning: Failed to create employee_notes table: %v", err)
	}

	// Insert some test notes
	_, err = localDB.Exec(`
		INSERT IGNORE INTO employee_notes (emp_id, note) VALUES
		(1, 'Great performance this quarter'),
		(2, 'Completed certification program'),
		(3, 'Leading new project initiative'),
		(1, 'Promoted to senior position'),
		(4, 'Excellent teamwork')
	`)

	// Test JOIN query
	fmt.Println("\n=== Testing JOIN between cached employees and local employee_notes ===")
	
	// Register employees as cached
	handler.SQLParser.RegisterCachedTable("testdb", "employees")
	
	// Build join query dynamically based on the ID column
	joinQuery := fmt.Sprintf(`
		SELECT 
			e.%s as emp_id,
			n.note,
			n.created_at
		FROM employees e
		JOIN employee_notes n ON e.%s = n.emp_id
		LIMIT 10
	`, idColumn, idColumn)

	fmt.Printf("\nExecuting JOIN query:\n%s\n", joinQuery)

	result2, err := handler.ExecuteQuery(joinQuery, "testdb")
	if err != nil {
		fmt.Printf("Join query failed: %v\n", err)
		
		// Try a simpler approach - just show both tables separately
		fmt.Println("\nShowing data from both tables separately:")
		
		// Cached employees
		result, _ := handler.ExecuteQuery("SELECT * FROM employees LIMIT 3", "testdb")
		if result != nil {
			fmt.Println("\nEmployees (from cache):")
			for i, row := range result.Rows {
				fmt.Printf("  %v\n", row)
				if i >= 2 {
					break
				}
			}
		}
		
		// Local notes
		rows, _ := localDB.Query("SELECT * FROM employee_notes LIMIT 3")
		if rows != nil {
			fmt.Println("\nEmployee Notes (from local):")
			for rows.Next() {
				var noteId, empId int
				var note string
				var createdAt sql.NullString
				rows.Scan(&noteId, &empId, &note, &createdAt)
				fmt.Printf("  Note %d: Employee %d - %s\n", noteId, empId, note)
			}
			rows.Close()
		}
	} else if result2 != nil {
		fmt.Printf("\nJoin successful! Results:\n")
		fmt.Printf("Columns: %v\n", result2.Columns)
		for i, row := range result2.Rows {
			fmt.Printf("Employee %v: %v (created: %v)\n", row[0], row[1], row[2])
			if i >= 4 {
				fmt.Printf("... (%d more rows)\n", len(result2.Rows)-5)
				break
			}
		}
	}

	// Clean up test table
	localDB.Exec("DROP TABLE IF EXISTS employee_notes")
	
	fmt.Println("\n=== Test Complete ===")
}