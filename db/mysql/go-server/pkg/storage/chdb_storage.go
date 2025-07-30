package storage

import (
	"fmt"
	"strings"
	"sync"

	"github.com/chdb-io/chdb-go/chdb"
	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/types"
	"github.com/rs/zerolog"
)

// ChDBStorage implements the Storage interface using chDB
type ChDBStorage struct {
	session *chdb.Session
	dbPath  string
	logger  zerolog.Logger
	mutex   sync.RWMutex
	
	// Track which tables are stored in chDB
	tables map[string]map[string]bool // database -> table -> exists
}

// NewChDBStorage creates a new chDB storage instance
func NewChDBStorage(dbPath string, logger zerolog.Logger) (*ChDBStorage, error) {
	// Create chDB session
	session, err := chdb.NewSession(dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create chDB session: %w", err)
	}
	
	storage := &ChDBStorage{
		session: session,
		dbPath:  dbPath,
		logger:  logger,
		tables:  make(map[string]map[string]bool),
	}
	
	// Initialize default database
	if err := storage.initializeChDB(); err != nil {
		session.Close()
		return nil, fmt.Errorf("failed to initialize chDB: %w", err)
	}
	
	return storage, nil
}

// initializeChDB creates the initial database structure
func (s *ChDBStorage) initializeChDB() error {
	queries := []string{
		"CREATE DATABASE IF NOT EXISTS default",
		"CREATE DATABASE IF NOT EXISTS system",
	}
	
	for _, query := range queries {
		s.mutex.Lock()
		_, err := s.session.Query(query, "CSV")
		s.mutex.Unlock()
		if err != nil {
			return fmt.Errorf("failed to execute init query: %w", err)
		}
	}
	
	return nil
}

// Close closes the chDB session
func (s *ChDBStorage) Close() error {
	if s.session != nil {
		s.session.Close()
	}
	return nil
}

// CreateDatabase creates a new database
func (s *ChDBStorage) CreateDatabase(name string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	query := fmt.Sprintf("CREATE DATABASE IF NOT EXISTS %s", name)
	_, err := s.session.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("failed to create database %s: %w", name, err)
	}
	
	// Initialize tables map for this database
	if s.tables[name] == nil {
		s.tables[name] = make(map[string]bool)
	}
	
	s.logger.Info().Str("database", name).Msg("Created database in chDB")
	return nil
}

// DropDatabase drops a database
func (s *ChDBStorage) DropDatabase(name string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	query := fmt.Sprintf("DROP DATABASE IF EXISTS %s", name)
	_, err := s.session.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("failed to drop database %s: %w", name, err)
	}
	
	// Remove from tables map
	delete(s.tables, name)
	
	s.logger.Info().Str("database", name).Msg("Dropped database from chDB")
	return nil
}

// HasDatabase checks if a database exists
func (s *ChDBStorage) HasDatabase(name string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := "SELECT name FROM system.databases WHERE name = ?"
	result, err := s.session.Query(strings.ReplaceAll(query, "?", fmt.Sprintf("'%s'", name)), "CSV")
	if err != nil || result == nil {
		return false
	}
	
	return result.Len() > 0
}

// GetDatabaseNames returns all database names
func (s *ChDBStorage) GetDatabaseNames() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := "SELECT name FROM system.databases"
	result, err := s.session.Query(query, "CSV")
	if err != nil || result == nil {
		return []string{}
	}
	
	var names []string
	data := result.String()
	lines := strings.Split(strings.TrimSpace(data), "\n")
	for _, line := range lines {
		name := strings.Trim(line, "\"")
		if name != "" && name != "system" && name != "INFORMATION_SCHEMA" {
			names = append(names, name)
		}
	}
	
	return names
}

// CreateTable creates a table in chDB with MergeTree engine
func (s *ChDBStorage) CreateTable(database, tableName string, schema sql.Schema) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	// Build CREATE TABLE query
	var columns []string
	var orderBy []string
	
	for _, col := range schema {
		colDef := fmt.Sprintf("%s %s", col.Name, s.sqlTypeToChDBType(col.Type))
		if !col.Nullable {
			colDef += " NOT NULL"
		}
		columns = append(columns, colDef)
		
		// Use primary key columns for ORDER BY
		if col.PrimaryKey {
			orderBy = append(orderBy, col.Name)
		}
	}
	
	// If no primary key, use first column for ordering
	if len(orderBy) == 0 && len(schema) > 0 {
		orderBy = append(orderBy, schema[0].Name)
	}
	
	query := fmt.Sprintf(
		"CREATE TABLE IF NOT EXISTS %s.%s (%s) ENGINE = MergeTree() ORDER BY (%s)",
		database, tableName,
		strings.Join(columns, ", "),
		strings.Join(orderBy, ", "),
	)
	
	_, err := s.session.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("failed to create table %s.%s: %w", database, tableName, err)
	}
	
	// Track table
	if s.tables[database] == nil {
		s.tables[database] = make(map[string]bool)
	}
	s.tables[database][tableName] = true
	
	s.logger.Info().
		Str("database", database).
		Str("table", tableName).
		Msg("Created table in chDB")
	
	return nil
}

// sqlTypeToChDBType converts go-mysql-server types to ClickHouse types
func (s *ChDBStorage) sqlTypeToChDBType(sqlType sql.Type) string {
	// Use string representation to determine type
	typeStr := sqlType.String()
	
	switch {
	case strings.Contains(typeStr, "TINYINT"), strings.Contains(typeStr, "SMALLINT"):
		return "Int16"
	case strings.Contains(typeStr, "MEDIUMINT"):
		return "Int32"
	case strings.Contains(typeStr, "BIGINT"):
		return "Int64"
	case strings.Contains(typeStr, "INT"):
		return "Int32"
	case strings.Contains(typeStr, "FLOAT"):
		return "Float32"
	case strings.Contains(typeStr, "DOUBLE"):
		return "Float64"
	case strings.Contains(typeStr, "DECIMAL"):
		return "Decimal(18, 4)"
	case strings.Contains(typeStr, "DATE") && !strings.Contains(typeStr, "DATETIME"):
		return "Date"
	case strings.Contains(typeStr, "DATETIME"), strings.Contains(typeStr, "TIMESTAMP"):
		return "DateTime"
	case strings.Contains(typeStr, "TEXT"), strings.Contains(typeStr, "BLOB"):
		return "String"
	case strings.Contains(typeStr, "BIT"):
		return "UInt8"
	default:
		// Default to String for varchar and other types
		return "String"
	}
}

// DropTable drops a table
func (s *ChDBStorage) DropTable(database, tableName string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	query := fmt.Sprintf("DROP TABLE IF EXISTS %s.%s", database, tableName)
	_, err := s.session.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("failed to drop table %s.%s: %w", database, tableName, err)
	}
	
	// Remove from tracking
	if s.tables[database] != nil {
		delete(s.tables[database], tableName)
	}
	
	s.logger.Info().
		Str("database", database).
		Str("table", tableName).
		Msg("Dropped table from chDB")
	
	return nil
}

// HasTable checks if a table exists
func (s *ChDBStorage) HasTable(database, tableName string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	if tables, ok := s.tables[database]; ok {
		return tables[tableName]
	}
	
	// Double check with chDB
	query := fmt.Sprintf(
		"SELECT name FROM system.tables WHERE database = '%s' AND name = '%s'",
		database, tableName,
	)
	result, err := s.session.Query(query, "CSV")
	if err != nil || result == nil {
		return false
	}
	
	return result.Len() > 0
}

// GetTableNames returns all table names in a database
func (s *ChDBStorage) GetTableNames(database string) []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := fmt.Sprintf("SELECT name FROM system.tables WHERE database = '%s'", database)
	result, err := s.session.Query(query, "CSV")
	if err != nil || result == nil {
		return []string{}
	}
	
	var names []string
	data := result.String()
	lines := strings.Split(strings.TrimSpace(data), "\n")
	for _, line := range lines {
		name := strings.Trim(line, "\"")
		if name != "" {
			names = append(names, name)
		}
	}
	
	return names
}

// GetTableSchema returns the schema for a table
func (s *ChDBStorage) GetTableSchema(database, tableName string) (sql.Schema, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := fmt.Sprintf(
		"SELECT name, type FROM system.columns WHERE database = '%s' AND table = '%s' ORDER BY position",
		database, tableName,
	)
	result, err := s.session.Query(query, "CSV")
	if err != nil || result == nil {
		return nil, fmt.Errorf("table %s.%s not found", database, tableName)
	}
	
	var schema sql.Schema
	data := result.String()
	lines := strings.Split(strings.TrimSpace(data), "\n")
	
	for i, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		
		colName := strings.Trim(parts[0], "\"")
		colType := strings.Trim(parts[1], "\"")
		
		// Convert chDB type back to SQL type
		sqlType := s.chDBTypeToSQLType(colType)
		
		col := &sql.Column{
			Name:     colName,
			Type:     sqlType,
			Nullable: strings.Contains(colType, "Nullable"),
			Source:   tableName,
			PrimaryKey: i == 0, // Assume first column is primary key for simplicity
		}
		
		schema = append(schema, col)
	}
	
	return schema, nil
}

// chDBTypeToSQLType converts ClickHouse types back to SQL types
func (s *ChDBStorage) chDBTypeToSQLType(chType string) sql.Type {
	// Remove Nullable wrapper if present
	chType = strings.TrimPrefix(chType, "Nullable(")
	chType = strings.TrimSuffix(chType, ")")
	
	switch {
	case strings.HasPrefix(chType, "Int8"):
		return types.Int8
	case strings.HasPrefix(chType, "Int16"):
		return types.Int16
	case strings.HasPrefix(chType, "Int32"):
		return types.Int32
	case strings.HasPrefix(chType, "Int64"):
		return types.Int64
	case strings.HasPrefix(chType, "UInt"):
		return types.Uint32
	case strings.HasPrefix(chType, "Float32"):
		return types.Float32
	case strings.HasPrefix(chType, "Float64"):
		return types.Float64
	case strings.HasPrefix(chType, "Decimal"):
		return types.MustCreateDecimalType(18, 4)
	case chType == "Date":
		return types.Date
	case chType == "DateTime":
		return types.Datetime
	default:
		return types.Text
	}
}

// InsertRow inserts a row into a table
func (s *ChDBStorage) InsertRow(database, tableName string, row sql.Row) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	// Build values string
	var values []string
	for _, val := range row {
		values = append(values, s.formatValue(val))
	}
	
	query := fmt.Sprintf(
		"INSERT INTO %s.%s VALUES (%s)",
		database, tableName,
		strings.Join(values, ", "),
	)
	
	_, err := s.session.Query(query, "CSV")
	if err != nil {
		return fmt.Errorf("failed to insert row: %w", err)
	}
	
	return nil
}

// formatValue formats a value for chDB insertion
func (s *ChDBStorage) formatValue(val interface{}) string {
	if val == nil {
		return "NULL"
	}
	
	switch v := val.(type) {
	case string:
		// Escape single quotes
		escaped := strings.ReplaceAll(v, "'", "\\'")
		return fmt.Sprintf("'%s'", escaped)
	case []byte:
		// Convert bytes to string
		escaped := strings.ReplaceAll(string(v), "'", "\\'")
		return fmt.Sprintf("'%s'", escaped)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// UpdateRow is not directly supported in chDB - we'll delete and insert
func (s *ChDBStorage) UpdateRow(database, tableName string, oldRow, newRow sql.Row) error {
	// chDB doesn't support UPDATE efficiently
	// For analytical workloads, this is rarely needed
	// Could implement with ALTER TABLE UPDATE if needed
	return fmt.Errorf("UPDATE not supported in chDB storage - use delete + insert")
}

// DeleteRow is not directly supported in chDB
func (s *ChDBStorage) DeleteRow(database, tableName string, row sql.Row) error {
	// chDB doesn't support DELETE efficiently
	// For analytical workloads, this is rarely needed
	// Could implement with ALTER TABLE DELETE if needed
	return fmt.Errorf("DELETE not supported in chDB storage - use table recreation")
}

// GetRows returns all rows from a table
func (s *ChDBStorage) GetRows(database, tableName string) ([]sql.Row, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := fmt.Sprintf("SELECT * FROM %s.%s", database, tableName)
	result, err := s.session.Query(query, "CSV")
	if err != nil {
		return nil, fmt.Errorf("failed to get rows: %w", err)
	}
	
	if result == nil || result.Len() == 0 {
		return []sql.Row{}, nil
	}
	
	// Parse CSV result
	var rows []sql.Row
	data := result.String()
	lines := strings.Split(strings.TrimSpace(data), "\n")
	
	for _, line := range lines {
		if line == "" {
			continue
		}
		
		// Simple CSV parsing - would need proper parser for production
		values := s.parseCSVLine(line)
		row := make(sql.Row, len(values))
		for i, val := range values {
			row[i] = val
		}
		rows = append(rows, row)
	}
	
	return rows, nil
}

// parseCSVLine does simple CSV parsing
func (s *ChDBStorage) parseCSVLine(line string) []string {
	var result []string
	var current strings.Builder
	inQuotes := false
	
	for i := 0; i < len(line); i++ {
		ch := line[i]
		
		if ch == '"' && (i == 0 || line[i-1] != '\\') {
			inQuotes = !inQuotes
		} else if ch == ',' && !inQuotes {
			result = append(result, current.String())
			current.Reset()
		} else {
			current.WriteByte(ch)
		}
	}
	
	if current.Len() > 0 {
		result = append(result, current.String())
	}
	
	return result
}

// ExecuteQuery executes a raw query on chDB - useful for analytical queries
func (s *ChDBStorage) ExecuteQuery(query string) ([][]interface{}, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	result, err := s.session.Query(query, "CSV")
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	
	if result == nil || result.Len() == 0 {
		return [][]interface{}{}, nil
	}
	
	// Parse result
	var rows [][]interface{}
	data := result.String()
	lines := strings.Split(strings.TrimSpace(data), "\n")
	
	for _, line := range lines {
		if line == "" {
			continue
		}
		
		values := s.parseCSVLine(line)
		row := make([]interface{}, len(values))
		for i, val := range values {
			row[i] = val
		}
		rows = append(rows, row)
	}
	
	return rows, nil
}

// Ensure we implement the Storage interface
var _ Storage = (*ChDBStorage)(nil)