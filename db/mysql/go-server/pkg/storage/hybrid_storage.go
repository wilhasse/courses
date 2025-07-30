package storage

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/dolthub/go-mysql-server/sql"
	"github.com/rs/zerolog"
)

// TableMetadata tracks information about tables for routing decisions
type TableMetadata struct {
	RowCount       int64
	SizeBytes      int64
	LastAccessed   time.Time
	AccessCount    int64
	IsAnalytical   bool  // True if table is used for analytical queries
	StorageBackend string // "lmdb", "chdb", or "remote"
}

// HybridStorage implements intelligent routing between LMDB, chDB, and remote MySQL
type HybridStorage struct {
	lmdb      *LMDBStorage
	chdb      *ChDBStorage
	logger    zerolog.Logger
	
	// Table metadata for routing decisions
	metadata  map[string]map[string]*TableMetadata // database -> table -> metadata
	metaMutex sync.RWMutex
	
	// Configuration thresholds
	hotDataThreshold    int64 // Row count threshold for LMDB (default 1M rows)
	analyticalThreshold int64 // Row count threshold for chDB (default 10M rows)
	
	// Track which backend stores each table
	tableBackends map[string]map[string]string // database -> table -> backend
}

// NewHybridStorage creates a new hybrid storage instance
func NewHybridStorage(lmdbPath, chdbPath string, logger zerolog.Logger) (*HybridStorage, error) {
	// Initialize LMDB for hot data
	lmdbStorage, err := NewLMDBStorage(lmdbPath, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create LMDB storage: %w", err)
	}
	
	// Initialize chDB for analytical data
	chdbStorage, err := NewChDBStorage(chdbPath, logger)
	if err != nil {
		lmdbStorage.Close()
		return nil, fmt.Errorf("failed to create chDB storage: %w", err)
	}
	
	return &HybridStorage{
		lmdb:                lmdbStorage,
		chdb:                chdbStorage,
		logger:              logger,
		metadata:            make(map[string]map[string]*TableMetadata),
		hotDataThreshold:    1_000_000,  // 1M rows for LMDB
		analyticalThreshold: 10_000_000, // 10M rows for chDB
		tableBackends:       make(map[string]map[string]string),
	}, nil
}

// Close closes all storage backends
func (s *HybridStorage) Close() error {
	var errs []error
	
	if err := s.lmdb.Close(); err != nil {
		errs = append(errs, fmt.Errorf("lmdb close error: %w", err))
	}
	
	if err := s.chdb.Close(); err != nil {
		errs = append(errs, fmt.Errorf("chdb close error: %w", err))
	}
	
	if len(errs) > 0 {
		return fmt.Errorf("close errors: %v", errs)
	}
	
	return nil
}

// getBackendForTable determines which backend should handle a table
func (s *HybridStorage) getBackendForTable(database, table string) Storage {
	s.metaMutex.RLock()
	defer s.metaMutex.RUnlock()
	
	// Check if we have a specific backend assignment
	if backends, ok := s.tableBackends[database]; ok {
		if backend, ok := backends[table]; ok {
			switch backend {
			case "chdb":
				return s.chdb
			case "lmdb":
				return s.lmdb
			}
		}
	}
	
	// Check metadata for routing decision
	if dbMeta, ok := s.metadata[database]; ok {
		if tableMeta, ok := dbMeta[table]; ok {
			// Route based on table characteristics
			if tableMeta.IsAnalytical || tableMeta.RowCount > s.analyticalThreshold {
				s.logger.Debug().
					Str("database", database).
					Str("table", table).
					Str("backend", "chdb").
					Int64("rows", tableMeta.RowCount).
					Msg("Routing to chDB (analytical)")
				return s.chdb
			}
			
			if tableMeta.RowCount < s.hotDataThreshold && tableMeta.AccessCount > 10 {
				s.logger.Debug().
					Str("database", database).
					Str("table", table).
					Str("backend", "lmdb").
					Int64("rows", tableMeta.RowCount).
					Msg("Routing to LMDB (hot data)")
				return s.lmdb
			}
		}
	}
	
	// Default to LMDB for small tables
	return s.lmdb
}

// markTableForBackend assigns a table to a specific backend
func (s *HybridStorage) markTableForBackend(database, table, backend string) {
	s.metaMutex.Lock()
	defer s.metaMutex.Unlock()
	
	if s.tableBackends[database] == nil {
		s.tableBackends[database] = make(map[string]string)
	}
	s.tableBackends[database][table] = backend
}

// CreateDatabase creates a database in the appropriate backend
func (s *HybridStorage) CreateDatabase(name string) error {
	// Create in both backends to allow flexibility
	if err := s.lmdb.CreateDatabase(name); err != nil {
		return err
	}
	
	if err := s.chdb.CreateDatabase(name); err != nil {
		// Rollback LMDB creation
		s.lmdb.DropDatabase(name)
		return err
	}
	
	// Initialize metadata
	s.metaMutex.Lock()
	s.metadata[name] = make(map[string]*TableMetadata)
	s.tableBackends[name] = make(map[string]string)
	s.metaMutex.Unlock()
	
	return nil
}

// DropDatabase drops a database from all backends
func (s *HybridStorage) DropDatabase(name string) error {
	var errs []error
	
	if err := s.lmdb.DropDatabase(name); err != nil {
		errs = append(errs, err)
	}
	
	if err := s.chdb.DropDatabase(name); err != nil {
		errs = append(errs, err)
	}
	
	// Clean up metadata
	s.metaMutex.Lock()
	delete(s.metadata, name)
	delete(s.tableBackends, name)
	s.metaMutex.Unlock()
	
	if len(errs) > 0 {
		return fmt.Errorf("drop database errors: %v", errs)
	}
	
	return nil
}

// HasDatabase checks if a database exists in any backend
func (s *HybridStorage) HasDatabase(name string) bool {
	return s.lmdb.HasDatabase(name) || s.chdb.HasDatabase(name)
}

// GetDatabaseNames returns all database names from all backends
func (s *HybridStorage) GetDatabaseNames() []string {
	nameSet := make(map[string]bool)
	
	// Get from LMDB
	for _, name := range s.lmdb.GetDatabaseNames() {
		nameSet[name] = true
	}
	
	// Get from chDB
	for _, name := range s.chdb.GetDatabaseNames() {
		nameSet[name] = true
	}
	
	// Convert to slice
	var names []string
	for name := range nameSet {
		names = append(names, name)
	}
	
	return names
}

// CreateTable creates a table in the appropriate backend based on hints
func (s *HybridStorage) CreateTable(database, tableName string, schema sql.Schema) error {
	// Analyze schema to determine best backend
	backend := s.analyzeSchemaForBackend(database, tableName, schema)
	
	s.logger.Info().
		Str("database", database).
		Str("table", tableName).
		Str("backend", backend).
		Msg("Creating table in selected backend")
	
	// Create in the selected backend
	var err error
	switch backend {
	case "chdb":
		err = s.chdb.CreateTable(database, tableName, schema)
		s.markTableForBackend(database, tableName, "chdb")
	default:
		err = s.lmdb.CreateTable(database, tableName, schema)
		s.markTableForBackend(database, tableName, "lmdb")
	}
	
	if err != nil {
		return err
	}
	
	// Initialize metadata
	s.metaMutex.Lock()
	if s.metadata[database] == nil {
		s.metadata[database] = make(map[string]*TableMetadata)
	}
	s.metadata[database][tableName] = &TableMetadata{
		RowCount:       0,
		LastAccessed:   time.Now(),
		AccessCount:    0,
		IsAnalytical:   backend == "chdb",
		StorageBackend: backend,
	}
	s.metaMutex.Unlock()
	
	return nil
}

// analyzeSchemaForBackend determines the best backend based on schema
func (s *HybridStorage) analyzeSchemaForBackend(database, table string, schema sql.Schema) string {
	// Heuristics for analytical tables
	analyticalKeywords := []string{
		"fact", "events", "logs", "metrics", "analytics",
		"transactions", "orders", "sales", "impressions",
	}
	
	tableLower := strings.ToLower(table)
	for _, keyword := range analyticalKeywords {
		if strings.Contains(tableLower, keyword) {
			return "chdb"
		}
	}
	
	// Tables with many numeric columns are likely analytical
	numericCount := 0
	for _, col := range schema {
		typeStr := col.Type.String()
		if strings.Contains(typeStr, "INT") || strings.Contains(typeStr, "FLOAT") || 
		   strings.Contains(typeStr, "DOUBLE") || strings.Contains(typeStr, "DECIMAL") {
			numericCount++
		}
	}
	
	if float64(numericCount)/float64(len(schema)) > 0.6 {
		return "chdb"
	}
	
	// Default to LMDB for transactional tables
	return "lmdb"
}

// DropTable drops a table from the appropriate backend
func (s *HybridStorage) DropTable(database, tableName string) error {
	backend := s.getBackendForTable(database, tableName)
	err := backend.DropTable(database, tableName)
	
	// Clean up metadata
	s.metaMutex.Lock()
	if s.metadata[database] != nil {
		delete(s.metadata[database], tableName)
	}
	if s.tableBackends[database] != nil {
		delete(s.tableBackends[database], tableName)
	}
	s.metaMutex.Unlock()
	
	return err
}

// HasTable checks if a table exists in any backend
func (s *HybridStorage) HasTable(database, tableName string) bool {
	return s.lmdb.HasTable(database, tableName) || s.chdb.HasTable(database, tableName)
}

// GetTableNames returns all table names from all backends
func (s *HybridStorage) GetTableNames(database string) []string {
	tableSet := make(map[string]bool)
	
	// Get from LMDB
	for _, name := range s.lmdb.GetTableNames(database) {
		tableSet[name] = true
	}
	
	// Get from chDB
	for _, name := range s.chdb.GetTableNames(database) {
		tableSet[name] = true
	}
	
	// Convert to slice
	var names []string
	for name := range tableSet {
		names = append(names, name)
	}
	
	return names
}

// GetTableSchema returns the schema from the appropriate backend
func (s *HybridStorage) GetTableSchema(database, tableName string) (sql.Schema, error) {
	backend := s.getBackendForTable(database, tableName)
	return backend.GetTableSchema(database, tableName)
}

// InsertRow inserts a row into the appropriate backend
func (s *HybridStorage) InsertRow(database, tableName string, row sql.Row) error {
	backend := s.getBackendForTable(database, tableName)
	err := backend.InsertRow(database, tableName, row)
	
	// Update metadata
	s.updateTableMetadata(database, tableName, 1)
	
	return err
}

// UpdateRow updates a row in the appropriate backend
func (s *HybridStorage) UpdateRow(database, tableName string, oldRow, newRow sql.Row) error {
	backend := s.getBackendForTable(database, tableName)
	
	// chDB doesn't support updates well, so we might need to handle differently
	if backend == s.chdb {
		s.logger.Warn().
			Str("database", database).
			Str("table", tableName).
			Msg("UPDATE on chDB table - performance may be impacted")
	}
	
	return backend.UpdateRow(database, tableName, oldRow, newRow)
}

// DeleteRow deletes a row from the appropriate backend
func (s *HybridStorage) DeleteRow(database, tableName string, row sql.Row) error {
	backend := s.getBackendForTable(database, tableName)
	
	// chDB doesn't support deletes well, so we might need to handle differently
	if backend == s.chdb {
		s.logger.Warn().
			Str("database", database).
			Str("table", tableName).
			Msg("DELETE on chDB table - performance may be impacted")
	}
	
	err := backend.DeleteRow(database, tableName, row)
	
	// Update metadata
	s.updateTableMetadata(database, tableName, -1)
	
	return err
}

// GetRows returns all rows from the appropriate backend
func (s *HybridStorage) GetRows(database, tableName string) ([]sql.Row, error) {
	backend := s.getBackendForTable(database, tableName)
	
	// Update access metadata
	s.metaMutex.Lock()
	if s.metadata[database] != nil && s.metadata[database][tableName] != nil {
		s.metadata[database][tableName].LastAccessed = time.Now()
		s.metadata[database][tableName].AccessCount++
	}
	s.metaMutex.Unlock()
	
	return backend.GetRows(database, tableName)
}

// updateTableMetadata updates table statistics
func (s *HybridStorage) updateTableMetadata(database, table string, rowDelta int64) {
	s.metaMutex.Lock()
	defer s.metaMutex.Unlock()
	
	if s.metadata[database] == nil {
		s.metadata[database] = make(map[string]*TableMetadata)
	}
	
	if s.metadata[database][table] == nil {
		s.metadata[database][table] = &TableMetadata{}
	}
	
	meta := s.metadata[database][table]
	meta.RowCount += rowDelta
	meta.LastAccessed = time.Now()
	
	// Consider migration if table has grown significantly
	if meta.StorageBackend == "lmdb" && meta.RowCount > s.analyticalThreshold {
		s.logger.Info().
			Str("database", database).
			Str("table", table).
			Int64("rows", meta.RowCount).
			Msg("Table should be migrated to chDB for better performance")
	}
}

// MigrateTableToChDB migrates a table from LMDB to chDB
func (s *HybridStorage) MigrateTableToChDB(database, table string) error {
	s.logger.Info().
		Str("database", database).
		Str("table", table).
		Msg("Migrating table from LMDB to chDB")
	
	// Get schema from LMDB
	schema, err := s.lmdb.GetTableSchema(database, table)
	if err != nil {
		return fmt.Errorf("failed to get schema: %w", err)
	}
	
	// Get all rows from LMDB
	rows, err := s.lmdb.GetRows(database, table)
	if err != nil {
		return fmt.Errorf("failed to get rows: %w", err)
	}
	
	// Create table in chDB
	if err := s.chdb.CreateTable(database, table, schema); err != nil {
		return fmt.Errorf("failed to create table in chDB: %w", err)
	}
	
	// Insert rows in batches
	batchSize := 10000
	for i := 0; i < len(rows); i += batchSize {
		end := i + batchSize
		if end > len(rows) {
			end = len(rows)
		}
		
		for _, row := range rows[i:end] {
			if err := s.chdb.InsertRow(database, table, row); err != nil {
				// Rollback
				s.chdb.DropTable(database, table)
				return fmt.Errorf("failed to insert row: %w", err)
			}
		}
		
		s.logger.Debug().
			Str("database", database).
			Str("table", table).
			Int("progress", end).
			Int("total", len(rows)).
			Msg("Migration progress")
	}
	
	// Drop from LMDB
	if err := s.lmdb.DropTable(database, table); err != nil {
		s.logger.Warn().Err(err).Msg("Failed to drop table from LMDB after migration")
	}
	
	// Update metadata
	s.markTableForBackend(database, table, "chdb")
	
	s.logger.Info().
		Str("database", database).
		Str("table", table).
		Int("rows", len(rows)).
		Msg("Table migration completed")
	
	return nil
}

// GetTableMetadata returns metadata for a table
func (s *HybridStorage) GetTableMetadata(database, table string) *TableMetadata {
	s.metaMutex.RLock()
	defer s.metaMutex.RUnlock()
	
	if s.metadata[database] != nil {
		return s.metadata[database][table]
	}
	
	return nil
}

// SetAnalyticalTable marks a table as analytical (should use chDB)
func (s *HybridStorage) SetAnalyticalTable(database, table string, analytical bool) {
	s.metaMutex.Lock()
	defer s.metaMutex.Unlock()
	
	if s.metadata[database] == nil {
		s.metadata[database] = make(map[string]*TableMetadata)
	}
	
	if s.metadata[database][table] == nil {
		s.metadata[database][table] = &TableMetadata{}
	}
	
	s.metadata[database][table].IsAnalytical = analytical
}

// SetThresholds updates the thresholds for storage backend selection
func (s *HybridStorage) SetThresholds(hotDataThreshold, analyticalThreshold int64) {
	s.hotDataThreshold = hotDataThreshold
	s.analyticalThreshold = analyticalThreshold
	
	s.logger.Info().
		Int64("hotDataThreshold", hotDataThreshold).
		Int64("analyticalThreshold", analyticalThreshold).
		Msg("Updated hybrid storage thresholds")
}

// Ensure we implement the Storage interface
var _ Storage = (*HybridStorage)(nil)