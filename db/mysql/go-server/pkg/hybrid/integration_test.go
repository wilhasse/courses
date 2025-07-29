package hybrid

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"testing"

	_ "github.com/go-sql-driver/mysql"
	"github.com/rs/zerolog"
)

// TestHybridQuery demonstrates the hybrid query functionality
func TestHybridQuery(t *testing.T) {
	// Skip if not in integration test mode
	if os.Getenv("INTEGRATION_TEST") != "true" {
		t.Skip("Skipping integration test. Set INTEGRATION_TEST=true to run")
	}

	// Configure logger
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	// MySQL connection string
	// Update these values to match your MySQL setup
	mysqlDSN := "root:password@tcp(localhost:3306)/testdb"

	// Create hybrid handler
	config := Config{
		MySQLDSN: mysqlDSN,
		LMDBPath: "/tmp/hybrid_test_lmdb",
		Logger:   logger,
	}

	handler, err := NewHybridHandler(config)
	if err != nil {
		t.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer handler.Close()

	// Clean up LMDB directory after test
	defer os.RemoveAll("/tmp/hybrid_test_lmdb")

	// Load ACORDO_GM table into LMDB
	t.Run("LoadTable", func(t *testing.T) {
		err := handler.LoadTable("testdb", "ACORDO_GM")
		if err != nil {
			t.Fatalf("Failed to load ACORDO_GM table: %v", err)
		}

		// Verify table is cached
		if !handler.IsTableCached("testdb", "ACORDO_GM") {
			t.Error("ACORDO_GM table should be cached")
		}
	})

	// Test simple SELECT from cached table
	t.Run("SimpleSelect", func(t *testing.T) {
		query := "SELECT * FROM ACORDO_GM LIMIT 10"
		result, err := handler.ExecuteQuery(query, "testdb")
		if err != nil {
			t.Fatalf("Failed to execute query: %v", err)
		}

		if result == nil {
			t.Fatal("Expected result, got nil")
		}

		t.Logf("Query returned %d rows with %d columns", len(result.Rows), len(result.Columns))
		
		// Print first few rows
		for i, row := range result.Rows {
			if i >= 3 {
				break
			}
			t.Logf("Row %d: %v", i, row)
		}
	})

	// Test JOIN between cached and remote tables
	t.Run("JoinQuery", func(t *testing.T) {
		// Assuming there's another table that references ACORDO_GM
		query := `
			SELECT a.*, b.* 
			FROM ACORDO_GM a 
			JOIN other_table b ON a.id = b.acordo_id 
			LIMIT 10
		`
		
		result, err := handler.ExecuteQuery(query, "testdb")
		if err != nil {
			// This might fail if other_table doesn't exist
			t.Logf("Join query failed (expected if other_table doesn't exist): %v", err)
			return
		}

		if result != nil {
			t.Logf("Join query returned %d rows", len(result.Rows))
		}
	})

	// Test query analysis
	t.Run("QueryAnalysis", func(t *testing.T) {
		queries := []string{
			"SELECT * FROM ACORDO_GM",
			"SELECT * FROM non_cached_table",
			"SELECT a.*, b.* FROM ACORDO_GM a JOIN other_table b ON a.id = b.id",
			"SELECT * FROM ACORDO_GM WHERE status = 'ACTIVE'",
		}

		for _, query := range queries {
			analysis, err := handler.sqlParser.AnalyzeQuery(query, "testdb")
			if err != nil {
				t.Errorf("Failed to analyze query '%s': %v", query, err)
				continue
			}

			t.Logf("Query: %s", query)
			t.Logf("  Has cached table: %v", analysis.HasCachedTable)
			t.Logf("  Cached tables: %v", analysis.CachedTables)
			t.Logf("  Remote tables: %v", analysis.RemoteTables)
			t.Logf("  Is join query: %v", analysis.IsJoinQuery)
			t.Logf("  Requires rewrite: %v", analysis.RequiresRewrite)
		}
	})

	// Test query rewriting
	t.Run("QueryRewriting", func(t *testing.T) {
		query := "SELECT a.id, a.name, b.description FROM ACORDO_GM a JOIN other_table b ON a.id = b.acordo_id"
		
		rewriteResult, err := handler.queryRewriter.RewriteQuery(query, "testdb")
		if err != nil {
			t.Fatalf("Failed to rewrite query: %v", err)
		}

		t.Logf("Original query: %s", query)
		t.Logf("Remote query: %s", rewriteResult.RemoteQuery)
		t.Logf("Cached tables: %v", rewriteResult.CachedTableNames)
		t.Logf("Join strategy: %+v", rewriteResult.JoinStrategy)
	})

	// Test performance comparison
	t.Run("PerformanceComparison", func(t *testing.T) {
		// Direct MySQL query
		mysqlConn, err := sql.Open("mysql", mysqlDSN)
		if err != nil {
			t.Fatalf("Failed to connect to MySQL: %v", err)
		}
		defer mysqlConn.Close()

		query := "SELECT COUNT(*) FROM ACORDO_GM"

		// Execute directly on MySQL
		var count int
		err = mysqlConn.QueryRow(query).Scan(&count)
		if err != nil {
			t.Logf("Direct MySQL query failed: %v", err)
		} else {
			t.Logf("ACORDO_GM has %d rows", count)
		}

		// Execute through hybrid handler
		result, err := handler.ExecuteQuery(query, "testdb")
		if err != nil {
			t.Logf("Hybrid query failed: %v", err)
		} else if result != nil && len(result.Rows) > 0 {
			t.Logf("Hybrid query returned count: %v", result.Rows[0][0])
		}
	})
}

// Example demonstrates how to use the hybrid query system
func Example() {
	// Configure logger
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	// Create hybrid handler
	config := Config{
		MySQLDSN: "root:password@tcp(localhost:3306)/mydb",
		LMDBPath: "/var/lib/mysql-hybrid/cache",
		Logger:   logger,
	}

	handler, err := NewHybridHandler(config)
	if err != nil {
		log.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer handler.Close()

	// Load ACORDO_GM table into LMDB cache
	err = handler.LoadTable("mydb", "ACORDO_GM")
	if err != nil {
		log.Fatalf("Failed to load table: %v", err)
	}

	// Execute a query that joins cached and remote tables
	query := `
		SELECT 
			a.id, 
			a.agreement_number,
			a.status,
			t.transaction_date,
			t.amount
		FROM ACORDO_GM a
		JOIN transactions t ON a.id = t.acordo_id
		WHERE a.status = 'ACTIVE'
		AND t.transaction_date >= '2024-01-01'
		ORDER BY t.transaction_date DESC
		LIMIT 100
	`

	result, err := handler.ExecuteQuery(query, "mydb")
	if err != nil {
		log.Fatalf("Failed to execute query: %v", err)
	}

	// Process results
	fmt.Printf("Query returned %d rows\n", len(result.Rows))
	for i, row := range result.Rows {
		if i >= 5 {
			break // Show only first 5 rows
		}
		fmt.Printf("Row %d: %v\n", i+1, row)
	}
}

// BenchmarkHybridQuery benchmarks the hybrid query system
func BenchmarkHybridQuery(b *testing.B) {
	if os.Getenv("INTEGRATION_TEST") != "true" {
		b.Skip("Skipping benchmark. Set INTEGRATION_TEST=true to run")
	}

	logger := zerolog.New(os.Stdout).Level(zerolog.ErrorLevel)

	config := Config{
		MySQLDSN: "root:password@tcp(localhost:3306)/testdb",
		LMDBPath: "/tmp/hybrid_bench_lmdb",
		Logger:   logger,
	}

	handler, err := NewHybridHandler(config)
	if err != nil {
		b.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer handler.Close()
	defer os.RemoveAll("/tmp/hybrid_bench_lmdb")

	// Load table
	err = handler.LoadTable("testdb", "ACORDO_GM")
	if err != nil {
		b.Fatalf("Failed to load table: %v", err)
	}

	// Benchmark simple SELECT
	b.Run("SimpleSelect", func(b *testing.B) {
		query := "SELECT * FROM ACORDO_GM WHERE id = 12345"
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := handler.ExecuteQuery(query, "testdb")
			if err != nil {
				b.Fatalf("Query failed: %v", err)
			}
		}
	})

	// Benchmark JOIN query
	b.Run("JoinQuery", func(b *testing.B) {
		query := "SELECT a.*, b.* FROM ACORDO_GM a JOIN other_table b ON a.id = b.acordo_id LIMIT 100"
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := handler.ExecuteQuery(query, "testdb")
			if err != nil {
				// Skip if other_table doesn't exist
				b.Skip("Join query not available")
			}
		}
	})
}