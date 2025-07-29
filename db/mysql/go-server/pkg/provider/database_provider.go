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
	storage         storage.Storage
	databases       map[string]sql.Database // Changed to interface to support both Database and RemoteDatabase
	remoteDatabases map[string]*RemoteDatabase // Track remote databases separately for cleanup
	mu              sync.RWMutex
}

// NewDatabaseProvider creates a new database provider
func NewDatabaseProvider(storage storage.Storage) *DatabaseProvider {
	provider := &DatabaseProvider{
		storage:         storage,
		databases:       make(map[string]sql.Database),
		remoteDatabases: make(map[string]*RemoteDatabase),
	}

	// Load existing databases from storage
	provider.loadExistingDatabases()

	return provider
}

// loadExistingDatabases loads databases that already exist in storage
func (p *DatabaseProvider) loadExistingDatabases() {
	// Get list of databases from storage
	databaseNames := p.storage.GetDatabaseNames()
	
	for _, dbName := range databaseNames {
		key := strings.ToLower(dbName)
		// Only load if not already in memory
		if _, exists := p.databases[key]; !exists {
			db := NewDatabase(dbName, p.storage)
			p.databases[key] = db
		}
	}
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

	// Create database in storage first
	if err := p.storage.CreateDatabase(name); err != nil {
		return err
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

	// Check if it's a remote database
	if remoteDb, isRemote := p.remoteDatabases[key]; isRemote {
		// Close the remote connection
		if err := remoteDb.Close(); err != nil {
			return fmt.Errorf("failed to close remote database connection: %v", err)
		}
		delete(p.remoteDatabases, key)
		delete(p.databases, key)
		return nil
	}

	// For local databases, drop all tables
	if db, ok := p.databases[key].(*Database); ok {
		tables, _ := db.GetTableNames(ctx)
		for _, tableName := range tables {
			if err := p.storage.DropTable(name, tableName); err != nil {
				return fmt.Errorf("failed to drop table %s: %v", tableName, err)
			}
		}
	}

	delete(p.databases, key)
	return nil
}

// CreateRemoteDatabase creates a new remote database connection
func (p *DatabaseProvider) CreateRemoteDatabase(ctx *sql.Context, name string, config RemoteConfig) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := p.databases[key]; exists {
		return sql.ErrDatabaseExists.New(name)
	}

	// Create remote database connection
	remoteDb, err := NewRemoteDatabase(name, config)
	if err != nil {
		return fmt.Errorf("failed to create remote database: %v", err)
	}

	p.databases[key] = remoteDb
	p.remoteDatabases[key] = remoteDb

	return nil
}

// Ensure we implement the required interfaces
var _ sql.DatabaseProvider = (*DatabaseProvider)(nil)
var _ sql.MutableDatabaseProvider = (*DatabaseProvider)(nil)