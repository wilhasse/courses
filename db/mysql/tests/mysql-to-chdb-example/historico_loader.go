package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/chdb-io/chdb-go/chdb"
	_ "github.com/go-sql-driver/mysql"
)

const (
	ROWS_PER_CHUNK  = 50000
	BATCH_SIZE      = 5000
)

type Loader struct {
	mysqlDB     *sql.DB
	chdbSession *chdb.Session
	startTime   time.Time
	totalLoaded int64
}

func NewLoader() *Loader {
	return &Loader{
		startTime: time.Now(),
	}
}

func (l *Loader) connectMySQL(host, user, password, database string) error {
	dsn := fmt.Sprintf("%s:%s@tcp(%s)/%s", user, password, host, database)
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return fmt.Errorf("failed to open MySQL connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return fmt.Errorf("failed to ping MySQL: %w", err)
	}

	l.mysqlDB = db
	log.Printf("[%s] Connected to MySQL successfully", time.Now().Format("2006-01-02 15:04:05"))
	return nil
}

func (l *Loader) initChDB(chdbPath string) error {
	session, err := chdb.NewSession(chdbPath)
	if err != nil {
		return fmt.Errorf("failed to create chdb session: %w", err)
	}

	l.chdbSession = session
	log.Printf("chdb session initialized with path: %s", chdbPath)
	return nil
}

func (l *Loader) createTables() error {
	queries := []string{
		"CREATE DATABASE IF NOT EXISTS mysql_import",
		`CREATE TABLE IF NOT EXISTS mysql_import.historico (
			id_contr Int32, seq UInt16, id_funcionario Int32,
			id_tel Int32, data DateTime, codigo UInt16, modo String
		) ENGINE = MergeTree() ORDER BY (id_contr, seq)`,
		`CREATE TABLE IF NOT EXISTS mysql_import.historico_texto (
			id_contr Int32, seq UInt16, mensagem String,
			motivo String, autorizacao String
		) ENGINE = MergeTree() ORDER BY (id_contr, seq)`,
	}

	for _, query := range queries {
		result, err := l.chdbSession.Query(query, "CSV")
		if err != nil {
			return fmt.Errorf("failed to execute query %s: %w", query, err)
		}
		// Check if result contains error by examining the output
		if result != nil && strings.Contains(result.String(), "Code:") {
			return fmt.Errorf("query error: %s", result.String())
		}
	}

	log.Println("Tables created successfully")
	return nil
}

func (l *Loader) getTotalRows() (int64, error) {
	var count int64
	err := l.mysqlDB.QueryRow("SELECT COUNT(*) FROM HISTORICO").Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get row count: %w", err)
	}
	return count, nil
}

func escapeString(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\\", "\\\\"), "'", "\\'")
}

func (l *Loader) loadData(totalRows int64, startOffset int64) error {
	// Stop merges before bulk loading
	log.Printf("[%s] Stopping merges for bulk loading...", time.Now().Format("2006-01-02 15:04:05"))
	result, err := l.chdbSession.Query("SYSTEM STOP MERGES mysql_import.historico", "CSV")
	if err != nil {
		log.Printf("Warning: failed to stop merges: %v", err)
	}
	if result != nil && strings.Contains(result.String(), "Code:") {
		log.Printf("Warning: stop merges error: %s", result.String())
	}

	offset := startOffset
	l.totalLoaded = startOffset
	chunkNumber := int(startOffset / ROWS_PER_CHUNK)
	totalChunks := int((totalRows + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK)

	for offset < totalRows {
		chunkNumber++
		chunkStart := time.Now()
		
		log.Printf("\n[%s] Processing chunk %d/%d (rows %d-%d of %d)...",
			time.Now().Format("2006-01-02 15:04:05"),
			chunkNumber, totalChunks,
			offset, min(offset+ROWS_PER_CHUNK, totalRows), totalRows)

		// Query MySQL data
		query := fmt.Sprintf(`
			SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO
			FROM HISTORICO
			ORDER BY ID_CONTR, SEQ
			LIMIT %d OFFSET %d`, ROWS_PER_CHUNK, offset)

		rows, err := l.mysqlDB.Query(query)
		if err != nil {
			log.Printf("MySQL query error: %v", err)
			offset += ROWS_PER_CHUNK
			continue
		}

		// Process rows in batches
		var batch []string
		batchCount := 0
		chunkRows := 0

		for rows.Next() {
			var idContr, idFunc, idTel int
			var seq, codigo int
			var data, modo string

			err := rows.Scan(&idContr, &seq, &idFunc, &idTel, &data, &codigo, &modo)
			if err != nil {
				log.Printf("Row scan error: %v", err)
				continue
			}

			// Build value string
			value := fmt.Sprintf("(%d, %d, %d, %d, '%s', %d, '%s')",
				idContr, seq, idFunc, idTel, data, codigo, escapeString(modo))
			
			batch = append(batch, value)
			batchCount++
			chunkRows++

			// Insert when batch is full
			if batchCount == BATCH_SIZE {
				if err := l.insertBatch(batch); err != nil {
					log.Printf("Batch insert error: %v", err)
				}
				l.totalLoaded += int64(batchCount)
				batch = nil
				batchCount = 0

				// Progress update
				if l.totalLoaded%10000 == 0 {
					elapsed := time.Since(l.startTime).Seconds()
					rowsPerSec := float64(l.totalLoaded-startOffset) / elapsed
					log.Printf("  Progress: %d HISTORICO rows loaded (%.0f rows/sec)",
						l.totalLoaded, rowsPerSec)
				}
			}
		}
		rows.Close()

		// Insert remaining rows
		if batchCount > 0 {
			if err := l.insertBatch(batch); err != nil {
				log.Printf("Final batch insert error: %v", err)
			}
			l.totalLoaded += int64(batchCount)
		}

		// Chunk completion stats
		chunkDuration := time.Since(chunkStart).Seconds()
		elapsed := time.Since(l.startTime).Seconds()
		avgSpeed := float64(l.totalLoaded-startOffset) / elapsed
		
		log.Printf("  HISTORICO: %d rows loaded for this chunk", chunkRows)
		log.Printf("  [%s] Chunk %d completed in %.0f seconds (avg: %.0f rows/sec)",
			time.Now().Format("2006-01-02 15:04:05"),
			chunkNumber, chunkDuration, avgSpeed)

		// Periodic verification
		if chunkNumber%10 == 0 {
			l.verifyAndFlush()
		}

		// Progress estimate
		progress := float64(offset) / float64(totalRows) * 100
		remainingRows := totalRows - offset
		etaSeconds := float64(remainingRows) / avgSpeed
		log.Printf("  Progress: %.1f%% (%d/%d rows) - ETA: %.0f minutes",
			progress, l.totalLoaded, totalRows, etaSeconds/60)

		offset += ROWS_PER_CHUNK
	}

	// Re-enable merges and optimize
	log.Printf("\n[%s] Re-enabling merges and optimizing table...",
		time.Now().Format("2006-01-02 15:04:05"))
	
	queries := []string{
		"SYSTEM START MERGES mysql_import.historico",
		"OPTIMIZE TABLE mysql_import.historico FINAL",
	}
	
	for _, q := range queries {
		result, err := l.chdbSession.Query(q, "CSV")
		if err != nil {
			log.Printf("Error executing %s: %v", q, err)
		}
		if result != nil && strings.Contains(result.String(), "Code:") {
			log.Printf("Query error for %s: %s", q, result.String())
		}
	}

	return nil
}

func (l *Loader) insertBatch(batch []string) error {
	if len(batch) == 0 {
		return nil
	}

	query := fmt.Sprintf("INSERT INTO mysql_import.historico VALUES %s",
		strings.Join(batch, ", "))
	
	// Debug first few batches
	if l.totalLoaded < 100000 {
		log.Printf("    Executing batch insert: %d rows, query size: %d KB",
			len(batch), len(query)/1024)
	}

	result, err := l.chdbSession.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("batch insert failed: %w", err)
	}
	if result != nil && strings.Contains(result.String(), "Code:") {
		return fmt.Errorf("batch insert error: %s", result.String())
	}

	return nil
}

func (l *Loader) verifyAndFlush() {
	log.Printf("  [%s] Flushing data to disk...", time.Now().Format("2006-01-02 15:04:05"))
	
	// Flush logs
	result, err := l.chdbSession.Query("SYSTEM FLUSH LOGS", "CSV")
	if err != nil {
		log.Printf("Warning: flush error: %v", err)
	}
	if result != nil && strings.Contains(result.String(), "Code:") {
		log.Printf("Warning: flush error: %s", result.String())
	}

	// Verify count
	result, err = l.chdbSession.Query("SELECT COUNT(*) FROM mysql_import.historico", "CSV")
	if err != nil {
		log.Printf("Count query error: %v", err)
		return
	}
	if result != nil && !strings.Contains(result.String(), "Code:") && result.Len() > 0 {
		log.Printf("  Actual rows in table: %s (expected: %d)", 
			strings.TrimSpace(result.String()), l.totalLoaded)
	}
}

func (l *Loader) cleanup() {
	if l.mysqlDB != nil {
		l.mysqlDB.Close()
	}
	if l.chdbSession != nil {
		// Final verification
		result, err := l.chdbSession.Query("SELECT COUNT(*) FROM mysql_import.historico", "CSV")
		if err == nil && result != nil && !strings.Contains(result.String(), "Code:") {
			log.Printf("\nFinal HISTORICO count: %s", strings.TrimSpace(result.String()))
		}
		l.chdbSession.Cleanup()
	}
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func main() {
	var (
		host       = flag.String("host", "localhost", "MySQL host")
		user       = flag.String("user", "root", "MySQL user")
		password   = flag.String("password", "", "MySQL password")
		database   = flag.String("database", "", "MySQL database")
		rowCount   = flag.Int64("row-count", 0, "Total row count (skip COUNT query)")
		offset     = flag.Int64("offset", 0, "Start from this offset")
		skipTexto  = flag.Bool("skip-texto", false, "Skip HISTORICO_TEXTO table")
		chdbPath   = flag.String("chdb-path", "/tmp/chdb", "Path for chdb data storage")
	)
	flag.Parse()
	
	// Support positional arguments for backward compatibility
	args := flag.Args()
	if len(args) >= 4 {
		*host = args[0]
		*user = args[1]
		*password = args[2]
		*database = args[3]
	}

	if *database == "" {
		fmt.Println("Usage: historico_loader_go [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		fmt.Println("\nAlternative: historico_loader_go <host> <user> <password> <database> [flags]")
		fmt.Println("\nExample:")
		fmt.Println("  ./historico_loader_go -host localhost -user root -password pass -database mydb -chdb-path /data/chdb")
		fmt.Println("  ./historico_loader_go localhost root pass mydb -row-count 300266692 -chdb-path /data/chdb")
		log.Fatal("\nError: Database name is required")
	}

	loader := NewLoader()
	defer loader.cleanup()

	// Connect to MySQL
	if err := loader.connectMySQL(*host, *user, *password, *database); err != nil {
		log.Fatal(err)
	}

	// Initialize chdb
	if err := loader.initChDB(*chdbPath); err != nil {
		log.Fatal(err)
	}

	// Create tables
	if err := loader.createTables(); err != nil {
		log.Fatal(err)
	}
	
	if *skipTexto {
		log.Println("Skipping HISTORICO_TEXTO table as requested")
	}

	// Get total rows
	totalRows := *rowCount
	if totalRows == 0 {
		log.Println("Getting row count from HISTORICO table...")
		count, err := loader.getTotalRows()
		if err != nil {
			log.Fatal(err)
		}
		totalRows = count
	}

	log.Printf("Total rows to process: %d", totalRows)
	if *offset > 0 {
		log.Printf("Starting from offset: %d", *offset)
	}

	// Load data
	startTime := time.Now()
	if err := loader.loadData(totalRows, *offset); err != nil {
		log.Fatal(err)
	}

	// Final stats
	duration := time.Since(startTime).Seconds()
	processedRows := loader.totalLoaded - *offset
	log.Printf("\n[%s] Loading completed!", time.Now().Format("2006-01-02 15:04:05"))
	log.Printf("Total HISTORICO rows loaded: %d", loader.totalLoaded)
	log.Printf("Rows processed this session: %d in %.0f seconds", processedRows, duration)
	log.Printf("Average speed: %.0f rows/second", float64(processedRows)/duration)
}