package provider

import (
	"strings"
	"sync"

	"github.com/dolthub/go-mysql-server/sql"
	"mysql-server-example/pkg/storage"
)

// Database implements sql.Database and related interfaces
type Database struct {
	name    string
	storage storage.Storage
	tables  map[string]*Table
	mu      sync.RWMutex
}

// NewDatabase creates a new database
func NewDatabase(name string, storage storage.Storage) *Database {
	return &Database{
		name:    name,
		storage: storage,
		tables:  make(map[string]*Table),
	}
}

// Name implements sql.Database
func (db *Database) Name() string {
	return db.name
}

// GetTableInsensitive implements sql.Database
func (db *Database) GetTableInsensitive(ctx *sql.Context, tblName string) (sql.Table, bool, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Check in-memory cache first
	if table, exists := db.tables[strings.ToLower(tblName)]; exists {
		return table, true, nil
	}

	// Check if table exists in storage
	if db.storage.HasTable(db.name, tblName) {
		table, err := db.loadTableFromStorage(tblName)
		if err != nil {
			return nil, false, err
		}
		db.tables[strings.ToLower(tblName)] = table
		return table, true, nil
	}

	return nil, false, nil
}

// GetTableNames implements sql.Database
func (db *Database) GetTableNames(ctx *sql.Context) ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	// Get tables from storage
	return db.storage.GetTableNames(db.name), nil
}

// CreateTable implements sql.TableCreator
func (db *Database) CreateTable(ctx *sql.Context, name string, schema sql.PrimaryKeySchema, collation sql.CollationID, comment string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := db.tables[key]; exists {
		return sql.ErrTableAlreadyExists.New(name)
	}

	// Create table in storage
	if err := db.storage.CreateTable(db.name, name, schema.Schema); err != nil {
		return err
	}

	// Create table object
	table := NewTable(name, schema.Schema, db.storage, db.name)
	db.tables[key] = table

	return nil
}

// DropTable implements sql.TableDropper
func (db *Database) DropTable(ctx *sql.Context, name string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := db.tables[key]; !exists {
		return sql.ErrTableNotFound.New(name)
	}

	// Drop from storage
	if err := db.storage.DropTable(db.name, name); err != nil {
		return err
	}

	// Remove from cache
	delete(db.tables, key)
	return nil
}

// loadTableFromStorage loads a table from storage
func (db *Database) loadTableFromStorage(name string) (*Table, error) {
	schema, err := db.storage.GetTableSchema(db.name, name)
	if err != nil {
		return nil, err
	}

	return NewTable(name, schema, db.storage, db.name), nil
}

// Note: Sample data creation removed - now done via SQL initialization scripts

// Ensure we implement the required interfaces
var _ sql.Database = (*Database)(nil)
var _ sql.TableCreator = (*Database)(nil)
var _ sql.TableDropper = (*Database)(nil)