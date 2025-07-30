package storage

import (
	"github.com/dolthub/go-mysql-server/sql"
)

// Storage defines the interface for the custom storage backend
type Storage interface {
	// Database operations
	CreateDatabase(name string) error
	DropDatabase(name string) error
	HasDatabase(name string) bool
	GetDatabaseNames() []string

	// Table operations
	CreateTable(database, tableName string, schema sql.Schema) error
	DropTable(database, tableName string) error
	HasTable(database, tableName string) bool
	GetTableNames(database string) []string
	GetTableSchema(database, tableName string) (sql.Schema, error)

	// Row operations
	InsertRow(database, tableName string, row sql.Row) error
	UpdateRow(database, tableName string, oldRow, newRow sql.Row) error
	DeleteRow(database, tableName string, row sql.Row) error
	GetRows(database, tableName string) ([]sql.Row, error)

	// Lifecycle
	Close() error
}