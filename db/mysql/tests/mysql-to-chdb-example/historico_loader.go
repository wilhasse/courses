package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"strings"
	"strconv"
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
	appendMode  bool
	lastDate    *time.Time
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
		) ENGINE = MergeTree() 
		ORDER BY (id_contr, seq)`,
		`CREATE TABLE IF NOT EXISTS mysql_import.historico_texto (
			id_contr Int32, seq UInt16, mensagem String,
			motivo String, autorizacao String
		) ENGINE = MergeTree() 
		ORDER BY (id_contr, seq)`,
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

func (l *Loader) getTotalRows(tableName string) (int64, error) {
	var count int64
	var query string
	
	// In append mode, only count rows newer than last date
	if l.appendMode && l.lastDate != nil {
		if tableName == "HISTORICO_TEXTO" {
			// For HISTORICO_TEXTO, join with HISTORICO to filter by date
			query = fmt.Sprintf(`
				SELECT COUNT(*) 
				FROM %s t
				INNER JOIN HISTORICO h ON t.ID_CONTR = h.ID_CONTR AND t.SEQ = h.SEQ
				WHERE h.DATA > '%s'`, 
				tableName, l.lastDate.Format("2006-01-02 15:04:05"))
		} else {
			// For HISTORICO table
			query = fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE DATA > '%s'", 
				tableName, l.lastDate.Format("2006-01-02 15:04:05"))
		}
	} else {
		// Normal mode - just count all rows
		query = fmt.Sprintf("SELECT COUNT(*) FROM %s", tableName)
	}
	
	err := l.mysqlDB.QueryRow(query).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("failed to get row count: %w", err)
	}
	
	// Check if we have the composite index for efficient keyset pagination
	var indexCount int
	err = l.mysqlDB.QueryRow(`
		SELECT COUNT(*) FROM information_schema.STATISTICS 
		WHERE TABLE_SCHEMA = DATABASE() 
		AND TABLE_NAME = ? 
		AND INDEX_NAME = 'idx_contr_seq'
	`, tableName).Scan(&indexCount)
	
	if err == nil && indexCount == 0 {
		log.Printf("IMPORTANT: Consider creating this index for optimal performance:")
		log.Printf("  CREATE INDEX idx_contr_seq ON %s (ID_CONTR, SEQ);", tableName)
	}
	
	return count, nil
}

func (l *Loader) getLastDate() error {
	if !l.appendMode {
		return nil
	}
	
	log.Printf("Checking last imported date in ClickHouse...")
	
	// Query the latest date from HISTORICO table
	result, err := l.chdbSession.Query(
		"SELECT MAX(data) FROM mysql_import.historico", "TSV")
	
	if err != nil || result == nil {
		log.Printf("No existing data found, will import all records")
		return nil
	}
	
	dateStr := strings.TrimSpace(result.String())
	if dateStr == "" || dateStr == "\\N" {
		log.Printf("No existing data found, will import all records")
		return nil
	}
	
	// Parse the date
	lastDate, err := time.Parse("2006-01-02 15:04:05", dateStr)
	if err != nil {
		return fmt.Errorf("failed to parse last date %s: %w", dateStr, err)
	}
	
	l.lastDate = &lastDate
	log.Printf("Last imported date: %s", lastDate.Format("2006-01-02 15:04:05"))
	log.Printf("Will import records with DATA > %s", lastDate.Format("2006-01-02 15:04:05"))
	
	return nil
}

func escapeString(s string) string {
	return strings.ReplaceAll(strings.ReplaceAll(s, "\\", "\\\\"), "'", "\\'")
}

func (l *Loader) loadData(totalRows int64, startOffset int64, skipTexto, onlyTexto bool) error {
	if onlyTexto {
		return l.loadHistoricoTexto(totalRows, startOffset)
	}
	
	// Load HISTORICO first
	if !onlyTexto {
		if err := l.loadHistorico(totalRows, startOffset); err != nil {
			return err
		}
	}
	
	// Then load HISTORICO_TEXTO if not skipped
	if !skipTexto && !onlyTexto {
		// Reset for HISTORICO_TEXTO
		l.totalLoaded = 0
		log.Printf("\n[%s] Starting HISTORICO_TEXTO load...", time.Now().Format("2006-01-02 15:04:05"))
		
		// Get row count for HISTORICO_TEXTO
		textoRows, err := l.getTotalRows("HISTORICO_TEXTO")
		if err != nil {
			log.Printf("Warning: failed to get HISTORICO_TEXTO row count: %v", err)
			textoRows = totalRows // Assume same as HISTORICO
		}
		
		if err := l.loadHistoricoTexto(textoRows, 0); err != nil {
			return err
		}
	}
	
	return nil
}

func (l *Loader) loadHistorico(totalRows int64, startOffset int64) error {
	log.Printf("[%s] Loading HISTORICO table using keyset pagination", time.Now().Format("2006-01-02 15:04:05"))
	
	// Initialize keyset values
	var lastIDContr int32 = -1
	var lastSeq uint16 = 0
	
	// If resuming from offset or append mode, try to get from ClickHouse first
	if startOffset > 0 && !l.appendMode {
		log.Printf("Finding resume point for offset %d...", startOffset)
		
		// First try to get the last row from ClickHouse (much faster!)
		chResult, err := l.chdbSession.Query(`
			SELECT id_contr, seq 
			FROM mysql_import.historico 
			ORDER BY id_contr DESC, seq DESC 
			LIMIT 1
		`, "TSV")
		
		if err == nil && chResult != nil && chResult.Len() > 0 {
			// Parse the result
			fields := strings.Fields(chResult.String())
			if len(fields) >= 2 {
				if id, err := strconv.Atoi(fields[0]); err == nil {
					lastIDContr = int32(id)
				}
				if s, err := strconv.Atoi(fields[1]); err == nil {
					lastSeq = uint16(s)
				}
				log.Printf("Resuming from last ClickHouse row: ID_CONTR=%d, SEQ=%d", lastIDContr, lastSeq)
				
				// Get actual count from ClickHouse
				countResult, _ := l.chdbSession.Query("SELECT COUNT(*) FROM mysql_import.historico", "CSV")
				if countResult != nil {
					if count, err := strconv.ParseInt(strings.TrimSpace(countResult.String()), 10, 64); err == nil {
						l.totalLoaded = count
						log.Printf("Actual rows already loaded: %d", l.totalLoaded)
					}
				}
			}
		} else {
			// Fallback to MySQL (slow but works)
			log.Printf("ClickHouse query failed, falling back to MySQL (this will be slow)...")
			resumeQuery := fmt.Sprintf(`
				SELECT ID_CONTR, SEQ FROM HISTORICO 
				ORDER BY ID_CONTR, SEQ 
				LIMIT 1 OFFSET %d`, startOffset-1)
			
			row := l.mysqlDB.QueryRow(resumeQuery)
			err := row.Scan(&lastIDContr, &lastSeq)
			if err != nil {
				log.Printf("Warning: couldn't find resume point, starting from beginning: %v", err)
				lastIDContr = -1
				lastSeq = 0
				startOffset = 0
			} else {
				log.Printf("Resuming from ID_CONTR=%d, SEQ=%d", lastIDContr, lastSeq)
			}
		}
	}

	// Don't overwrite totalLoaded if we got it from ClickHouse
	if l.totalLoaded == 0 {
		l.totalLoaded = startOffset
	}
	chunkNumber := int(l.totalLoaded / ROWS_PER_CHUNK)
	totalChunks := int((totalRows + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK)

	for l.totalLoaded < totalRows {
		chunkNumber++
		chunkStart := time.Now()
		
		log.Printf("\n[%s] Processing chunk %d/%d (approx rows %d-%d of %d)...",
			time.Now().Format("2006-01-02 15:04:05"),
			chunkNumber, totalChunks,
			l.totalLoaded, min(l.totalLoaded+ROWS_PER_CHUNK, totalRows), totalRows)

		// Query MySQL data using keyset pagination
		var query string
		var queryArgs []interface{}
		
		if l.appendMode && l.lastDate != nil {
			// In append mode, also filter by date
			query = `
				SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO
				FROM HISTORICO
				WHERE DATA > ? AND (ID_CONTR > ? OR (ID_CONTR = ? AND SEQ > ?))
				ORDER BY ID_CONTR, SEQ
				LIMIT ?`
			queryArgs = []interface{}{l.lastDate.Format("2006-01-02 15:04:05"), 
				lastIDContr, lastIDContr, lastSeq, ROWS_PER_CHUNK}
		} else {
			// Normal mode
			query = `
				SELECT ID_CONTR, SEQ, ID_FUNCIONARIO, ID_TEL, DATA, CODIGO, MODO
				FROM HISTORICO
				WHERE (ID_CONTR > ? OR (ID_CONTR = ? AND SEQ > ?))
				ORDER BY ID_CONTR, SEQ
				LIMIT ?`
			queryArgs = []interface{}{lastIDContr, lastIDContr, lastSeq, ROWS_PER_CHUNK}
		}

		rows, err := l.mysqlDB.Query(query, queryArgs...)
		if err != nil {
			log.Printf("MySQL query error: %v", err)
			break // Can't continue with keyset pagination on error
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
			
			// Remember the last values for next keyset query
			lastIDContr = int32(idContr)
			lastSeq = uint16(seq)

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

				// Remove intermediate progress - only show chunk summary
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
		progress := float64(l.totalLoaded) / float64(totalRows) * 100
		remainingRows := totalRows - l.totalLoaded
		etaSeconds := float64(remainingRows) / avgSpeed
		log.Printf("  Progress: %.1f%% (%d/%d rows) - ETA: %.0f minutes",
			progress, l.totalLoaded, totalRows, etaSeconds/60)
		
		// Check if we got fewer rows than expected (might be near the end)
		if chunkRows < ROWS_PER_CHUNK {
			log.Printf("  Reached end of table (got %d rows, expected %d)", chunkRows, ROWS_PER_CHUNK)
			break
		}
	}

	// Final optimization - only if we actually inserted data
	if chunkNumber > 0 {
		if err := l.optimizeTable("historico", l.totalLoaded-startOffset); err != nil {
			log.Printf("Warning: optimization failed: %v", err)
		}
	} else {
		log.Printf("\n[%s] No new HISTORICO data to import", time.Now().Format("2006-01-02 15:04:05"))
	}

	return nil
}

func (l *Loader) loadHistoricoTexto(totalRows int64, startOffset int64) error {
	log.Printf("[%s] Loading HISTORICO_TEXTO table using keyset pagination", time.Now().Format("2006-01-02 15:04:05"))
	
	// Initialize keyset values
	var lastIDContr int32 = -1
	var lastSeq uint16 = 0
	
	// If resuming from offset, try to get from ClickHouse first
	if startOffset > 0 {
		log.Printf("Finding resume point for offset %d...", startOffset)
		
		// First try to get the last row from ClickHouse (much faster!)
		chResult, err := l.chdbSession.Query(`
			SELECT id_contr, seq 
			FROM mysql_import.historico_texto 
			ORDER BY id_contr DESC, seq DESC 
			LIMIT 1
		`, "TSV")
		
		if err == nil && chResult != nil && chResult.Len() > 0 {
			// Parse the result
			fields := strings.Fields(chResult.String())
			if len(fields) >= 2 {
				if id, err := strconv.Atoi(fields[0]); err == nil {
					lastIDContr = int32(id)
				}
				if s, err := strconv.Atoi(fields[1]); err == nil {
					lastSeq = uint16(s)
				}
				log.Printf("Resuming from last ClickHouse row: ID_CONTR=%d, SEQ=%d", lastIDContr, lastSeq)
				
				// Get actual count from ClickHouse
				countResult, _ := l.chdbSession.Query("SELECT COUNT(*) FROM mysql_import.historico_texto", "CSV")
				if countResult != nil {
					if count, err := strconv.ParseInt(strings.TrimSpace(countResult.String()), 10, 64); err == nil {
						l.totalLoaded = count
						log.Printf("Actual rows already loaded: %d", l.totalLoaded)
					}
				}
			}
		} else {
			// Fallback to MySQL
			log.Printf("ClickHouse query failed, falling back to MySQL...")
			resumeQuery := fmt.Sprintf(`
				SELECT ID_CONTR, SEQ FROM HISTORICO_TEXTO 
				ORDER BY ID_CONTR, SEQ 
				LIMIT 1 OFFSET %d`, startOffset-1)
			
			row := l.mysqlDB.QueryRow(resumeQuery)
			err := row.Scan(&lastIDContr, &lastSeq)
			if err != nil {
				log.Printf("Warning: couldn't find resume point, starting from beginning: %v", err)
				lastIDContr = -1
				lastSeq = 0
				startOffset = 0
			} else {
				log.Printf("Resuming from ID_CONTR=%d, SEQ=%d", lastIDContr, lastSeq)
			}
		}
	}

	// Don't overwrite totalLoaded if we got it from ClickHouse
	if l.totalLoaded == 0 {
		l.totalLoaded = startOffset
	}
	chunkNumber := int(l.totalLoaded / ROWS_PER_CHUNK)
	totalChunks := int((totalRows + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK)

	for l.totalLoaded < totalRows {
		chunkNumber++
		chunkStart := time.Now()
		
		log.Printf("\n[%s] Processing HISTORICO_TEXTO chunk %d/%d (approx rows %d-%d of %d)...",
			time.Now().Format("2006-01-02 15:04:05"),
			chunkNumber, totalChunks,
			l.totalLoaded, min(l.totalLoaded+ROWS_PER_CHUNK, totalRows), totalRows)

		// Query MySQL data using keyset pagination
		var query string
		var queryArgs []interface{}
		
		if l.appendMode && l.lastDate != nil {
			// In append mode, join with HISTORICO to filter by date
			query = `
				SELECT t.ID_CONTR, t.SEQ, t.MENSAGEM, t.MOTIVO, t.AUTORIZACAO
				FROM HISTORICO_TEXTO t
				INNER JOIN HISTORICO h ON t.ID_CONTR = h.ID_CONTR AND t.SEQ = h.SEQ
				WHERE h.DATA > ? AND (t.ID_CONTR > ? OR (t.ID_CONTR = ? AND t.SEQ > ?))
				ORDER BY t.ID_CONTR, t.SEQ
				LIMIT ?`
			queryArgs = []interface{}{l.lastDate.Format("2006-01-02 15:04:05"), 
				lastIDContr, lastIDContr, lastSeq, ROWS_PER_CHUNK}
		} else {
			// Normal mode
			query = `
				SELECT ID_CONTR, SEQ, MENSAGEM, MOTIVO, AUTORIZACAO
				FROM HISTORICO_TEXTO
				WHERE (ID_CONTR > ? OR (ID_CONTR = ? AND SEQ > ?))
				ORDER BY ID_CONTR, SEQ
				LIMIT ?`
			queryArgs = []interface{}{lastIDContr, lastIDContr, lastSeq, ROWS_PER_CHUNK}
		}

		rows, err := l.mysqlDB.Query(query, queryArgs...)
		if err != nil {
			log.Printf("MySQL query error: %v", err)
			break
		}

		// Process rows in batches
		var batch []string
		batchCount := 0
		chunkRows := 0

		for rows.Next() {
			var idContr int
			var seq int
			var mensagem, motivo, autorizacao sql.NullString

			err := rows.Scan(&idContr, &seq, &mensagem, &motivo, &autorizacao)
			if err != nil {
				log.Printf("Row scan error: %v", err)
				continue
			}
			
			// Remember the last values for next keyset query
			lastIDContr = int32(idContr)
			lastSeq = uint16(seq)

			// Build value string
			value := fmt.Sprintf("(%d, %d, '%s', '%s', '%s')",
				idContr, seq, 
				escapeString(mensagem.String),
				escapeString(motivo.String),
				escapeString(autorizacao.String))
			
			batch = append(batch, value)
			batchCount++
			chunkRows++

			// Insert when batch is full
			if batchCount == BATCH_SIZE {
				if err := l.insertTextoBatch(batch); err != nil {
					log.Printf("Batch insert error: %v", err)
				}
				l.totalLoaded += int64(batchCount)
				batch = nil
				batchCount = 0
			}
		}
		rows.Close()

		// Insert remaining rows
		if batchCount > 0 {
			if err := l.insertTextoBatch(batch); err != nil {
				log.Printf("Final batch insert error: %v", err)
			}
			l.totalLoaded += int64(batchCount)
		}

		// Chunk completion stats
		chunkDuration := time.Since(chunkStart).Seconds()
		elapsed := time.Since(l.startTime).Seconds()
		avgSpeed := float64(l.totalLoaded-startOffset) / elapsed
		
		log.Printf("  HISTORICO_TEXTO: %d rows loaded for this chunk", chunkRows)
		log.Printf("  [%s] Chunk %d completed in %.0f seconds (avg: %.0f rows/sec)",
			time.Now().Format("2006-01-02 15:04:05"),
			chunkNumber, chunkDuration, avgSpeed)

		// Periodic verification
		if chunkNumber%10 == 0 {
			l.verifyAndFlushTexto()
		}

		// Progress estimate
		progress := float64(l.totalLoaded) / float64(totalRows) * 100
		remainingRows := totalRows - l.totalLoaded
		etaSeconds := float64(remainingRows) / avgSpeed
		log.Printf("  Progress: %.1f%% (%d/%d rows) - ETA: %.0f minutes",
			progress, l.totalLoaded, totalRows, etaSeconds/60)
		
		// Check if we got fewer rows than expected (might be near the end)
		if chunkRows < ROWS_PER_CHUNK {
			log.Printf("  Reached end of table (got %d rows, expected %d)", chunkRows, ROWS_PER_CHUNK)
			break
		}
	}

	// Final optimization - only if we actually inserted data
	if chunkNumber > 0 {
		if err := l.optimizeTable("historico_texto", l.totalLoaded-startOffset); err != nil {
			log.Printf("Warning: optimization failed: %v", err)
		}
	} else {
		log.Printf("\n[%s] No new HISTORICO_TEXTO data to import", time.Now().Format("2006-01-02 15:04:05"))
	}

	return nil
}

func (l *Loader) insertTextoBatch(batch []string) error {
	if len(batch) == 0 {
		return nil
	}

	query := fmt.Sprintf("INSERT INTO mysql_import.historico_texto VALUES %s",
		strings.Join(batch, ", "))
	
	result, err := l.chdbSession.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("batch insert failed: %w", err)
	}
	if result != nil && strings.Contains(result.String(), "Code:") {
		return fmt.Errorf("batch insert error: %s", result.String())
	}

	return nil
}

func (l *Loader) verifyAndFlushTexto() {
	log.Printf("  [%s] Flushing HISTORICO_TEXTO data to disk...", time.Now().Format("2006-01-02 15:04:05"))
	
	// Flush logs
	result, err := l.chdbSession.Query("SYSTEM FLUSH LOGS", "CSV")
	if err != nil {
		log.Printf("Warning: flush error: %v", err)
	}
	if result != nil && strings.Contains(result.String(), "Code:") {
		log.Printf("Warning: flush error: %s", result.String())
	}

	// Verify count
	result, err = l.chdbSession.Query("SELECT COUNT(*) FROM mysql_import.historico_texto", "CSV")
	if err != nil {
		log.Printf("Count query error: %v", err)
		return
	}
	if result != nil && !strings.Contains(result.String(), "Code:") && result.Len() > 0 {
		log.Printf("  Actual rows in table: %s (expected: %d)", 
			strings.TrimSpace(result.String()), l.totalLoaded)
	}
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
	
	// Check storage and parts every 50 chunks (2.5M rows)
	if l.totalLoaded % 2500000 == 0 {
		l.checkStorageStatus()
	}
}

func (l *Loader) checkStorageStatus() {
	log.Printf("  [%s] Storage and compression status:", time.Now().Format("2006-01-02 15:04:05"))
	
	// Check parts and size
	result, err := l.chdbSession.Query(`
		SELECT 
			count() as parts,
			formatReadableSize(sum(bytes_on_disk)) as size,
			formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed,
			round(sum(data_compressed_bytes) / sum(data_uncompressed_bytes), 2) as ratio
		FROM system.parts 
		WHERE database='mysql_import' AND table='historico' AND active
	`, "TSV")
	
	if err == nil && result != nil && !strings.Contains(result.String(), "Code:") {
		log.Printf("  Storage: %s", strings.TrimSpace(result.String()))
	}
}

func (l *Loader) optimizeTable(tableName string, rowsInserted int64) error {
	// Skip optimization if no rows were inserted
	if rowsInserted == 0 {
		return nil
	}
	
	// Check current table state
	result, err := l.chdbSession.Query(fmt.Sprintf(`
		SELECT 
			count() as parts_count,
			sum(rows) as total_rows,
			max(rows) as max_part_rows
		FROM system.parts 
		WHERE database='mysql_import' AND table='%s' AND active
	`, tableName), "TSV")
	
	if err != nil {
		return fmt.Errorf("failed to check table state: %w", err)
	}
	
	// Parse the result
	var partsCount int64
	if result != nil && result.Len() > 0 {
		fields := strings.Fields(result.String())
		if len(fields) >= 1 {
			partsCount, _ = strconv.ParseInt(fields[0], 10, 64)
		}
	}
	
	// Determine optimization strategy
	var optimizeQuery string
	
	if rowsInserted < 10000 && partsCount < 100 {
		// For small appends with few parts, skip optimization
		log.Printf("\n[%s] Skipping optimization for %s (only %d new rows, %d parts)", 
			time.Now().Format("2006-01-02 15:04:05"), tableName, rowsInserted, partsCount)
		return nil
	} else if rowsInserted < 1000000 && partsCount < 1000 {
		// For medium appends, use partition-based optimization
		log.Printf("\n[%s] Running partition optimization for %s...", 
			time.Now().Format("2006-01-02 15:04:05"), tableName)
		
		// Find partitions with the most recent data
		partResult, _ := l.chdbSession.Query(fmt.Sprintf(`
			SELECT DISTINCT partition
			FROM system.parts
			WHERE database='mysql_import' AND table='%s' AND active
			ORDER BY max_date DESC
			LIMIT 3
		`, tableName), "TSV")
		
		if partResult != nil && partResult.Len() > 0 {
			partitions := strings.Split(strings.TrimSpace(partResult.String()), "\n")
			for _, partition := range partitions {
				if partition != "" {
					optimizeQuery = fmt.Sprintf("OPTIMIZE TABLE mysql_import.%s PARTITION %s", 
						tableName, partition)
					_, err = l.chdbSession.Query(optimizeQuery, "CSV")
					if err != nil {
						log.Printf("Warning: failed to optimize partition %s: %v", partition, err)
					}
				}
			}
			return nil
		}
		
		// Fallback to regular optimization
		optimizeQuery = fmt.Sprintf("OPTIMIZE TABLE mysql_import.%s", tableName)
	} else {
		// For large appends or many parts, use FINAL optimization
		log.Printf("\n[%s] Running final optimization for %s (inserted %d rows, %d parts)...", 
			time.Now().Format("2006-01-02 15:04:05"), tableName, rowsInserted, partsCount)
		optimizeQuery = fmt.Sprintf("OPTIMIZE TABLE mysql_import.%s FINAL", tableName)
	}
	
	// Run optimization
	startTime := time.Now()
	result, err = l.chdbSession.Query(optimizeQuery, "CSV")
	duration := time.Since(startTime).Seconds()
	
	if err != nil {
		return fmt.Errorf("optimization failed: %w", err)
	}
	if result != nil && strings.Contains(result.String(), "Code:") {
		return fmt.Errorf("optimization error: %s", result.String())
	}
	
	log.Printf("  Optimization completed in %.0f seconds", duration)
	
	// Check final state
	if partsCount > 10 {
		finalResult, _ := l.chdbSession.Query(fmt.Sprintf(`
			SELECT count() as parts_after
			FROM system.parts 
			WHERE database='mysql_import' AND table='%s' AND active
		`, tableName), "TSV")
		
		if finalResult != nil && finalResult.Len() > 0 {
			partsAfter, _ := strconv.ParseInt(strings.TrimSpace(finalResult.String()), 10, 64)
			log.Printf("  Parts reduced from %d to %d", partsCount, partsAfter)
		}
	}
	
	return nil
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
		// DO NOT call Cleanup() - it deletes all data!
		// Use Close() instead - it only removes temp directories
		l.chdbSession.Close()
		log.Printf("\nIMPORTANT: Data stored at: %s", l.chdbSession.Path())
		log.Printf("Session closed. Data preserved.")
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
		onlyTexto  = flag.Bool("only-texto", false, "Process only HISTORICO_TEXTO table")
		chdbPath   = flag.String("chdb-path", "/tmp/chdb", "Path for chdb data storage")
		appendMode = flag.Bool("append", false, "Append mode: only import records newer than last imported date")
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
		fmt.Println("  # Initial load")
		fmt.Println("  ./historico_loader_go -host localhost -user root -password pass -database mydb -chdb-path /data/chdb")
		fmt.Println("  ")
		fmt.Println("  # Append new records (incremental update)")
		fmt.Println("  ./historico_loader_go -host localhost -user root -password pass -database mydb -chdb-path /data/chdb -append")
		log.Fatal("\nError: Database name is required")
	}

	loader := NewLoader()
	loader.appendMode = *appendMode
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
	
	// Get last date if in append mode
	if err := loader.getLastDate(); err != nil {
		log.Fatal(err)
	}
	
	// Validate flags
	if *skipTexto && *onlyTexto {
		log.Fatal("Cannot use both -skip-texto and -only-texto flags")
	}
	
	if *appendMode && *offset > 0 {
		log.Fatal("Cannot use both -append and -offset flags")
	}
	
	if *skipTexto {
		log.Println("Skipping HISTORICO_TEXTO table as requested")
	}
	if *onlyTexto {
		log.Println("Processing only HISTORICO_TEXTO table as requested")
	}
	if *appendMode {
		if loader.lastDate != nil {
			log.Printf("APPEND MODE: Importing records with DATA > %s", 
				loader.lastDate.Format("2006-01-02 15:04:05"))
		} else {
			log.Println("APPEND MODE: No existing data found, importing all records")
		}
	}

	// Get total rows
	totalRows := *rowCount
	tableName := "HISTORICO"
	if *onlyTexto {
		tableName = "HISTORICO_TEXTO"
	}
	
	if totalRows == 0 {
		log.Printf("Getting row count from %s table...", tableName)
		count, err := loader.getTotalRows(tableName)
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
	if err := loader.loadData(totalRows, *offset, *skipTexto, *onlyTexto); err != nil {
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