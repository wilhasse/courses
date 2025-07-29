package provider

import (
	"database/sql"
	"fmt"
	"log"
	"strings"
	"sync"

	_ "github.com/go-sql-driver/mysql"
	gmssql "github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/expression"
	"github.com/dolthub/go-mysql-server/sql/types"
	"github.com/dolthub/vitess/go/sqltypes"
)

// RemoteDatabase represents a database that mirrors schema from a remote MySQL server
type RemoteDatabase struct {
	name         string
	remoteConn   *sql.DB
	remoteConfig RemoteConfig
	tables       map[string]*ProxyTable
	mu           sync.RWMutex
}

// RemoteConfig holds configuration for remote MySQL connection
type RemoteConfig struct {
	Host     string
	Port     int
	User     string
	Password string
	Database string
}

// NewRemoteDatabase creates a new remote database connection
func NewRemoteDatabase(name string, config RemoteConfig) (*RemoteDatabase, error) {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?parseTime=true",
		config.User, config.Password, config.Host, config.Port, config.Database)
	
	conn, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to remote MySQL: %w", err)
	}
	
	// Test the connection
	if err := conn.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping remote MySQL: %w", err)
	}
	
	return &RemoteDatabase{
		name:         name,
		remoteConn:   conn,
		remoteConfig: config,
		tables:       make(map[string]*ProxyTable),
	}, nil
}

// Name implements sql.Database
func (db *RemoteDatabase) Name() string {
	return db.name
}

// GetTableInsensitive implements sql.Database
func (db *RemoteDatabase) GetTableInsensitive(ctx *gmssql.Context, tblName string) (gmssql.Table, bool, error) {
	db.mu.RLock()
	
	// Check cache first
	if table, exists := db.tables[strings.ToLower(tblName)]; exists {
		db.mu.RUnlock()
		log.Printf("RemoteDatabase[%s]: Using cached table %s", db.name, tblName)
		return table, true, nil
	}
	db.mu.RUnlock()
	
	log.Printf("RemoteDatabase[%s]: Fetching schema for table %s from remote", db.name, tblName)
	
	// Fetch schema from remote database
	schema, err := db.fetchTableSchema(tblName)
	if err != nil {
		log.Printf("RemoteDatabase[%s]: Error fetching schema for %s: %v", db.name, tblName, err)
		return nil, false, err
	}
	
	if schema == nil {
		log.Printf("RemoteDatabase[%s]: Table %s not found on remote", db.name, tblName)
		return nil, false, nil
	}
	
	log.Printf("RemoteDatabase[%s]: Schema fetched for %s, %d columns", db.name, tblName, len(schema))
	
	// Create proxy table
	proxyTable := NewProxyTable(tblName, schema, db.remoteConn, db.remoteConfig.Database)
	
	db.mu.Lock()
	db.tables[strings.ToLower(tblName)] = proxyTable
	db.mu.Unlock()
	
	return proxyTable, true, nil
}

// GetTableNames implements sql.Database
func (db *RemoteDatabase) GetTableNames(ctx *gmssql.Context) ([]string, error) {
	query := `
		SELECT TABLE_NAME 
		FROM information_schema.TABLES 
		WHERE TABLE_SCHEMA = ? 
		AND TABLE_TYPE = 'BASE TABLE'
		ORDER BY TABLE_NAME
	`
	
	rows, err := db.remoteConn.Query(query, db.remoteConfig.Database)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch table names: %w", err)
	}
	defer rows.Close()
	
	var tables []string
	for rows.Next() {
		var tableName string
		if err := rows.Scan(&tableName); err != nil {
			return nil, err
		}
		tables = append(tables, tableName)
	}
	
	return tables, nil
}

// fetchTableSchema queries the remote database's information_schema to get table structure
func (db *RemoteDatabase) fetchTableSchema(tableName string) (gmssql.Schema, error) {
	query := `
		SELECT 
			COLUMN_NAME,
			DATA_TYPE,
			IS_NULLABLE,
			COLUMN_DEFAULT,
			COLUMN_TYPE,
			COLUMN_KEY,
			EXTRA
		FROM information_schema.COLUMNS
		WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
		ORDER BY ORDINAL_POSITION
	`
	
	rows, err := db.remoteConn.Query(query, db.remoteConfig.Database, tableName)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch schema: %w", err)
	}
	defer rows.Close()
	
	var schema gmssql.Schema
	hasRows := false
	
	for rows.Next() {
		hasRows = true
		var (
			columnName   string
			dataType     string
			isNullable   string
			columnDefault sql.NullString
			columnType   string
			columnKey    string
			extra        string
		)
		
		if err := rows.Scan(&columnName, &dataType, &isNullable, &columnDefault, &columnType, &columnKey, &extra); err != nil {
			return nil, err
		}
		
		// Convert MySQL types to go-mysql-server types
		sqlType := convertMySQLType(dataType, columnType)
		
		nullable := isNullable == "YES"
		isPrimary := columnKey == "PRI"
		
		col := &gmssql.Column{
			Name:          columnName,
			Type:          sqlType,
			Nullable:      nullable,
			PrimaryKey:    isPrimary,
			AutoIncrement: strings.Contains(extra, "auto_increment"),
		}
		
		if columnDefault.Valid {
			// Create a literal expression for the default value
			col.Default = &gmssql.ColumnDefaultValue{
				Expr: expression.NewLiteral(columnDefault.String, types.Text),
			}
		}
		
		schema = append(schema, col)
	}
	
	if !hasRows {
		return nil, nil // Table doesn't exist
	}
	
	return schema, nil
}

// convertMySQLType converts MySQL type strings to go-mysql-server types
func convertMySQLType(dataType, columnType string) gmssql.Type {
	switch strings.ToLower(dataType) {
	case "tinyint":
		if strings.Contains(columnType, "unsigned") {
			return types.Uint8
		}
		return types.Int8
	case "smallint":
		if strings.Contains(columnType, "unsigned") {
			return types.Uint16
		}
		return types.Int16
	case "mediumint", "int", "integer":
		if strings.Contains(columnType, "unsigned") {
			return types.Uint32
		}
		return types.Int32
	case "bigint":
		if strings.Contains(columnType, "unsigned") {
			return types.Uint64
		}
		return types.Int64
	case "float":
		return types.Float32
	case "double", "real":
		return types.Float64
	case "decimal", "numeric":
		// Parse precision and scale from columnType if needed
		return types.Float64 // Simplified for now
	case "char":
		// Extract length from columnType if available
		return types.MustCreateStringWithDefaults(sqltypes.Char, 255)
	case "varchar":
		// Extract length from columnType if available
		return types.MustCreateStringWithDefaults(sqltypes.VarChar, 255)
	case "text", "tinytext", "mediumtext", "longtext":
		return types.Text
	case "blob", "tinyblob", "mediumblob", "longblob":
		return types.Blob
	case "date":
		return types.Date
	case "time":
		return types.Time
	case "datetime":
		return types.Datetime
	case "timestamp":
		return types.Timestamp
	case "year":
		return types.Year
	case "bit":
		return types.Uint64
	case "json":
		return types.JSON
	case "boolean", "bool":
		return types.Boolean
	default:
		return types.Text // Default fallback
	}
}

// Close closes the remote database connection
func (db *RemoteDatabase) Close() error {
	if db.remoteConn != nil {
		return db.remoteConn.Close()
	}
	return nil
}

// Ensure RemoteDatabase implements sql.Database
var _ gmssql.Database = (*RemoteDatabase)(nil)