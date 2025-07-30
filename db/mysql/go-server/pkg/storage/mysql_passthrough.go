package storage

import (
	"database/sql"
	"fmt"
	"strings"
	"sync"

	_ "github.com/go-sql-driver/mysql"
	gmssql "github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/types"
	"github.com/dolthub/vitess/go/sqltypes"
	"github.com/rs/zerolog"
)

// MySQLPassthroughStorage implements Storage interface by forwarding all operations to MySQL
type MySQLPassthroughStorage struct {
	db     *sql.DB
	config MySQLConfig
	logger zerolog.Logger
	mutex  sync.RWMutex
}

// MySQLConfig holds MySQL connection configuration
type MySQLConfig struct {
	Host              string
	Port              int
	User              string
	Password          string
	Database          string // If empty, mirrors all databases
	MaxOpenConns      int
	MaxIdleConns      int
	ConnMaxLifetime   string
}

// NewMySQLPassthroughStorage creates a new MySQL passthrough storage
func NewMySQLPassthroughStorage(config MySQLConfig, logger zerolog.Logger) (*MySQLPassthroughStorage, error) {
	// Build DSN
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/", 
		config.User, config.Password, config.Host, config.Port)
	
	if config.Database != "" {
		dsn += config.Database
	}
	
	// Add connection parameters
	dsn += "?parseTime=true&multiStatements=true"
	
	// Open connection
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open MySQL connection: %w", err)
	}
	
	// Configure connection pool
	if config.MaxOpenConns > 0 {
		db.SetMaxOpenConns(config.MaxOpenConns)
	}
	if config.MaxIdleConns > 0 {
		db.SetMaxIdleConns(config.MaxIdleConns)
	}
	
	// Test connection
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping MySQL: %w", err)
	}
	
	storage := &MySQLPassthroughStorage{
		db:     db,
		config: config,
		logger: logger,
	}
	
	logger.Info().
		Str("host", config.Host).
		Int("port", config.Port).
		Str("user", config.User).
		Msg("Connected to MySQL for passthrough storage")
	
	return storage, nil
}

// Close closes the MySQL connection
func (s *MySQLPassthroughStorage) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// CreateDatabase forwards to MySQL
func (s *MySQLPassthroughStorage) CreateDatabase(name string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	query := fmt.Sprintf("CREATE DATABASE IF NOT EXISTS `%s`", name)
	_, err := s.db.Exec(query)
	if err != nil {
		return fmt.Errorf("failed to create database: %w", err)
	}
	
	s.logger.Debug().Str("database", name).Msg("Created database in MySQL")
	return nil
}

// DropDatabase forwards to MySQL
func (s *MySQLPassthroughStorage) DropDatabase(name string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	query := fmt.Sprintf("DROP DATABASE IF EXISTS `%s`", name)
	_, err := s.db.Exec(query)
	if err != nil {
		return fmt.Errorf("failed to drop database: %w", err)
	}
	
	s.logger.Debug().Str("database", name).Msg("Dropped database from MySQL")
	return nil
}

// HasDatabase checks if database exists in MySQL
func (s *MySQLPassthroughStorage) HasDatabase(name string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	var exists string
	query := "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = ?"
	err := s.db.QueryRow(query, name).Scan(&exists)
	
	return err == nil && exists == name
}

// GetDatabaseNames returns all database names from MySQL
func (s *MySQLPassthroughStorage) GetDatabaseNames() []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA"
	rows, err := s.db.Query(query)
	if err != nil {
		s.logger.Error().Err(err).Msg("Failed to get database names")
		return []string{}
	}
	defer rows.Close()
	
	var databases []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err == nil {
			// Skip system databases unless explicitly configured
			if s.config.Database == "" {
				if name != "information_schema" && name != "mysql" && 
				   name != "performance_schema" && name != "sys" {
					databases = append(databases, name)
				}
			} else if name == s.config.Database {
				databases = append(databases, name)
			}
		}
	}
	
	return databases
}

// CreateTable forwards to MySQL
func (s *MySQLPassthroughStorage) CreateTable(database, tableName string, schema gmssql.Schema) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	// Build CREATE TABLE statement
	var columns []string
	var primaryKeys []string
	
	for _, col := range schema {
		colDef := fmt.Sprintf("`%s` %s", col.Name, s.sqlTypeToMySQL(col.Type))
		
		if !col.Nullable {
			colDef += " NOT NULL"
		}
		
		if col.AutoIncrement {
			colDef += " AUTO_INCREMENT"
		}
		
		if col.Default != nil {
			colDef += fmt.Sprintf(" DEFAULT %v", col.Default)
		}
		
		columns = append(columns, colDef)
		
		if col.PrimaryKey {
			primaryKeys = append(primaryKeys, fmt.Sprintf("`%s`", col.Name))
		}
	}
	
	if len(primaryKeys) > 0 {
		columns = append(columns, fmt.Sprintf("PRIMARY KEY (%s)", strings.Join(primaryKeys, ", ")))
	}
	
	query := fmt.Sprintf("CREATE TABLE IF NOT EXISTS `%s`.`%s` (%s)",
		database, tableName, strings.Join(columns, ", "))
	
	_, err := s.db.Exec(query)
	if err != nil {
		return fmt.Errorf("failed to create table: %w", err)
	}
	
	s.logger.Debug().
		Str("database", database).
		Str("table", tableName).
		Msg("Created table in MySQL")
	
	return nil
}

// sqlTypeToMySQL converts go-mysql-server types to MySQL types
func (s *MySQLPassthroughStorage) sqlTypeToMySQL(sqlType gmssql.Type) string {
	typeStr := strings.ToUpper(sqlType.String())
	
	// Handle common type conversions
	switch {
	case strings.Contains(typeStr, "VARCHAR"):
		if cs, ok := sqlType.(gmssql.StringType); ok {
			return fmt.Sprintf("VARCHAR(%d)", cs.MaxCharacterLength())
		}
		return "VARCHAR(255)"
	case strings.Contains(typeStr, "TEXT"):
		return "TEXT"
	case strings.Contains(typeStr, "INT"):
		if strings.Contains(typeStr, "UNSIGNED") {
			return strings.Replace(typeStr, "INT", "INT UNSIGNED", 1)
		}
		return typeStr
	case strings.Contains(typeStr, "DECIMAL"):
		if dt, ok := sqlType.(gmssql.DecimalType); ok {
			return fmt.Sprintf("DECIMAL(%d,%d)", dt.Precision(), dt.Scale())
		}
		return "DECIMAL(10,2)"
	default:
		return typeStr
	}
}

// DropTable forwards to MySQL
func (s *MySQLPassthroughStorage) DropTable(database, tableName string) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	query := fmt.Sprintf("DROP TABLE IF EXISTS `%s`.`%s`", database, tableName)
	_, err := s.db.Exec(query)
	if err != nil {
		return fmt.Errorf("failed to drop table: %w", err)
	}
	
	s.logger.Debug().
		Str("database", database).
		Str("table", tableName).
		Msg("Dropped table from MySQL")
	
	return nil
}

// HasTable checks if table exists in MySQL
func (s *MySQLPassthroughStorage) HasTable(database, tableName string) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	var exists string
	query := `SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
	          WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?`
	err := s.db.QueryRow(query, database, tableName).Scan(&exists)
	
	return err == nil && exists == tableName
}

// GetTableNames returns all table names in a database
func (s *MySQLPassthroughStorage) GetTableNames(database string) []string {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := `SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
	          WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'`
	rows, err := s.db.Query(query, database)
	if err != nil {
		s.logger.Error().Err(err).Msg("Failed to get table names")
		return []string{}
	}
	defer rows.Close()
	
	var tables []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err == nil {
			tables = append(tables, name)
		}
	}
	
	return tables
}

// GetTableSchema returns the schema for a table
func (s *MySQLPassthroughStorage) GetTableSchema(database, tableName string) (gmssql.Schema, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := `SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA, 
	                 CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
	          FROM INFORMATION_SCHEMA.COLUMNS 
	          WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
	          ORDER BY ORDINAL_POSITION`
	
	rows, err := s.db.Query(query, database, tableName)
	if err != nil {
		return nil, fmt.Errorf("failed to get schema: %w", err)
	}
	defer rows.Close()
	
	var schema gmssql.Schema
	for rows.Next() {
		var colName, dataType, isNullable, columnKey, extra string
		var charMaxLen, numPrecision, numScale sql.NullInt64
		
		err := rows.Scan(&colName, &dataType, &isNullable, &columnKey, &extra,
			&charMaxLen, &numPrecision, &numScale)
		if err != nil {
			continue
		}
		
		// Convert MySQL type to go-mysql-server type
		sqlType := s.mysqlTypeToSQL(dataType, charMaxLen.Int64, numPrecision.Int64, numScale.Int64)
		
		col := &gmssql.Column{
			Name:          colName,
			Type:          sqlType,
			Nullable:      isNullable == "YES",
			Source:        tableName,
			PrimaryKey:    strings.Contains(columnKey, "PRI"),
			AutoIncrement: strings.Contains(extra, "auto_increment"),
		}
		
		schema = append(schema, col)
	}
	
	if len(schema) == 0 {
		return nil, fmt.Errorf("table %s.%s not found", database, tableName)
	}
	
	return schema, nil
}

// mysqlTypeToSQL converts MySQL types to go-mysql-server types
func (s *MySQLPassthroughStorage) mysqlTypeToSQL(mysqlType string, charLen, precision, scale int64) gmssql.Type {
	mysqlType = strings.ToLower(mysqlType)
	
	switch mysqlType {
	case "tinyint":
		return types.Int8
	case "smallint":
		return types.Int16
	case "int", "integer":
		return types.Int32
	case "bigint":
		return types.Int64
	case "float":
		return types.Float32
	case "double":
		return types.Float64
	case "decimal":
		if precision > 0 {
			return types.MustCreateDecimalType(uint8(precision), uint8(scale))
		}
		return types.MustCreateDecimalType(10, 2)
	case "varchar":
		if charLen > 0 {
			return types.MustCreateStringWithDefaults(sqltypes.VarChar, charLen)
		}
		return types.MustCreateStringWithDefaults(sqltypes.VarChar, 255)
	case "text":
		return types.Text
	case "date":
		return types.Date
	case "datetime":
		return types.Datetime
	case "timestamp":
		return types.Timestamp
	case "blob":
		return types.Blob
	default:
		return types.Text
	}
}

// InsertRow forwards to MySQL
func (s *MySQLPassthroughStorage) InsertRow(database, tableName string, row gmssql.Row) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	// Get column names
	schema, err := s.GetTableSchema(database, tableName)
	if err != nil {
		return err
	}
	
	var columns []string
	var placeholders []string
	var values []interface{}
	
	for i, col := range schema {
		if i < len(row) {
			columns = append(columns, fmt.Sprintf("`%s`", col.Name))
			placeholders = append(placeholders, "?")
			values = append(values, row[i])
		}
	}
	
	query := fmt.Sprintf("INSERT INTO `%s`.`%s` (%s) VALUES (%s)",
		database, tableName,
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", "))
	
	_, err = s.db.Exec(query, values...)
	if err != nil {
		return fmt.Errorf("failed to insert row: %w", err)
	}
	
	return nil
}

// UpdateRow is complex for passthrough - would need to identify row by primary key
func (s *MySQLPassthroughStorage) UpdateRow(database, tableName string, oldRow, newRow gmssql.Row) error {
	// This is a simplified implementation
	// In a real implementation, you'd need to identify the row by primary key
	return fmt.Errorf("UPDATE not implemented for MySQL passthrough - use direct SQL")
}

// DeleteRow is complex for passthrough - would need to identify row by primary key  
func (s *MySQLPassthroughStorage) DeleteRow(database, tableName string, row gmssql.Row) error {
	// This is a simplified implementation
	// In a real implementation, you'd need to identify the row by primary key
	return fmt.Errorf("DELETE not implemented for MySQL passthrough - use direct SQL")
}

// GetRows returns all rows from a table
func (s *MySQLPassthroughStorage) GetRows(database, tableName string) ([]gmssql.Row, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	query := fmt.Sprintf("SELECT * FROM `%s`.`%s`", database, tableName)
	rows, err := s.db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to get rows: %w", err)
	}
	defer rows.Close()
	
	// Get column types
	columnTypes, err := rows.ColumnTypes()
	if err != nil {
		return nil, fmt.Errorf("failed to get column types: %w", err)
	}
	
	// Prepare scan destinations
	scanDests := make([]interface{}, len(columnTypes))
	for i := range scanDests {
		scanDests[i] = new(interface{})
	}
	
	var result []gmssql.Row
	for rows.Next() {
		err := rows.Scan(scanDests...)
		if err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}
		
		// Convert to gmssql.Row
		row := make(gmssql.Row, len(columnTypes))
		for i, dest := range scanDests {
			row[i] = *(dest.(*interface{}))
		}
		
		result = append(result, row)
	}
	
	return result, nil
}

// ExecuteQuery executes a raw query on MySQL (useful for complex operations)
func (s *MySQLPassthroughStorage) ExecuteQuery(query string) (*sql.Rows, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	return s.db.Query(query)
}

// Ensure we implement the Storage interface
var _ Storage = (*MySQLPassthroughStorage)(nil)