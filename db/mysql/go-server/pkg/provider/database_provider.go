package provider

import (
	"fmt"
	"strings"
	"sync"

	"github.com/dolthub/go-mysql-server/sql"
	"mysql-server-example/pkg/storage"
)

// DatabaseProvider implements sql.DatabaseProvider and sql.MutableDatabaseProvider
type DatabaseProvider struct {
	storage   storage.Storage
	databases map[string]*Database
	mu        sync.RWMutex
}

// NewDatabaseProvider creates a new database provider
func NewDatabaseProvider(storage storage.Storage) *DatabaseProvider {
	provider := &DatabaseProvider{
		storage:   storage,
		databases: make(map[string]*Database),
	}

	// Create a default database
	defaultDB := NewDatabase("testdb", storage)
	provider.databases["testdb"] = defaultDB

	// Add some sample tables
	defaultDB.CreateSampleTables()

	return provider
}

// Database implements sql.DatabaseProvider
func (p *DatabaseProvider) Database(ctx *sql.Context, name string) (sql.Database, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	db, exists := p.databases[strings.ToLower(name)]
	if !exists {
		return nil, sql.ErrDatabaseNotFound.New(name)
	}

	return db, nil
}

// HasDatabase implements sql.DatabaseProvider
func (p *DatabaseProvider) HasDatabase(ctx *sql.Context, name string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()

	_, exists := p.databases[strings.ToLower(name)]
	return exists
}

// AllDatabases implements sql.DatabaseProvider
func (p *DatabaseProvider) AllDatabases(ctx *sql.Context) []sql.Database {
	p.mu.RLock()
	defer p.mu.RUnlock()

	databases := make([]sql.Database, 0, len(p.databases))
	for _, db := range p.databases {
		databases = append(databases, db)
	}

	return databases
}

// CreateDatabase implements sql.MutableDatabaseProvider
func (p *DatabaseProvider) CreateDatabase(ctx *sql.Context, name string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := p.databases[key]; exists {
		return sql.ErrDatabaseExists.New(name)
	}

	// Create new database
	db := NewDatabase(name, p.storage)
	p.databases[key] = db

	return nil
}

// DropDatabase implements sql.MutableDatabaseProvider
func (p *DatabaseProvider) DropDatabase(ctx *sql.Context, name string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := p.databases[key]; !exists {
		return sql.ErrDatabaseNotFound.New(name)
	}

	// Drop all tables in the database
	db := p.databases[key]
	tables := db.GetTableNames(ctx)
	for _, tableName := range tables {
		if err := p.storage.DropTable(name, tableName); err != nil {
			return fmt.Errorf("failed to drop table %s: %v", tableName, err)
		}
	}

	delete(p.databases, key)
	return nil
}

// Ensure we implement the required interfaces
var _ sql.DatabaseProvider = (*DatabaseProvider)(nil)
var _ sql.MutableDatabaseProvider = (*DatabaseProvider)(nil)