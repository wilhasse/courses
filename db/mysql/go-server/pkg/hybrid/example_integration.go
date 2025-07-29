package hybrid

import (
	"context"
	"fmt"
	"io"
	"log"

	gmssql "github.com/dolthub/go-mysql-server/sql"
)

// HybridTable wraps a regular table with hybrid query capabilities
type HybridTable struct {
	name          string
	schema        gmssql.Schema
	handler       *HybridHandler
	database      string
	fallbackTable gmssql.Table // Original table for fallback
}

// NewHybridTable creates a new hybrid table wrapper
func NewHybridTable(name string, schema gmssql.Schema, handler *HybridHandler, database string, fallbackTable gmssql.Table) *HybridTable {
	return &HybridTable{
		name:          name,
		schema:        schema,
		handler:       handler,
		database:      database,
		fallbackTable: fallbackTable,
	}
}

// Name implements sql.Table
func (t *HybridTable) Name() string {
	return t.name
}

// String implements sql.Table
func (t *HybridTable) String() string {
	return t.name
}

// Schema implements sql.Table
func (t *HybridTable) Schema() gmssql.Schema {
	return t.schema
}

// Collation implements sql.Table
func (t *HybridTable) Collation() gmssql.CollationID {
	return gmssql.Collation_Default
}

// Partitions implements sql.Table
func (t *HybridTable) Partitions(ctx *gmssql.Context) (gmssql.PartitionIter, error) {
	// Check if this table is cached
	if t.handler.IsTableCached(t.database, t.name) {
		return &hybridPartitionIter{}, nil
	}
	
	// Fall back to original table
	return t.fallbackTable.Partitions(ctx)
}

// PartitionRows implements sql.Table
func (t *HybridTable) PartitionRows(ctx *gmssql.Context, partition gmssql.Partition) (gmssql.RowIter, error) {
	// Check if this table is cached
	if t.handler.IsTableCached(t.database, t.name) {
		// Get cached data
		rows, _, err := t.handler.GetCachedTableData(t.database, t.name)
		if err != nil {
			log.Printf("Failed to get cached data: %v", err)
			// Fall back to original table
			return t.fallbackTable.PartitionRows(ctx, partition)
		}
		
		return &hybridRowIter{
			rows:    rows,
			current: 0,
		}, nil
	}
	
	// Fall back to original table
	return t.fallbackTable.PartitionRows(ctx, partition)
}

// hybridPartitionIter implements sql.PartitionIter
type hybridPartitionIter struct {
	done bool
}

func (p *hybridPartitionIter) Next(ctx *gmssql.Context) (gmssql.Partition, error) {
	if p.done {
		return nil, io.EOF
	}
	p.done = true
	return &hybridPartition{}, nil
}

func (p *hybridPartitionIter) Close(ctx *gmssql.Context) error {
	return nil
}

// hybridPartition implements sql.Partition
type hybridPartition struct{}

func (p *hybridPartition) Key() []byte {
	return []byte("hybrid")
}

// hybridRowIter implements sql.RowIter
type hybridRowIter struct {
	rows    []gmssql.Row
	current int
}

func (r *hybridRowIter) Next(ctx *gmssql.Context) (gmssql.Row, error) {
	if r.current >= len(r.rows) {
		return nil, io.EOF
	}
	
	row := r.rows[r.current]
	r.current++
	return row, nil
}

func (r *hybridRowIter) Close(ctx *gmssql.Context) error {
	return nil
}

// Example of how to integrate with database provider
type HybridDatabaseProvider struct {
	originalProvider gmssql.DatabaseProvider
	hybridHandler    *HybridHandler
}

// NewHybridDatabaseProvider wraps an existing database provider with hybrid capabilities
func NewHybridDatabaseProvider(provider gmssql.DatabaseProvider, handler *HybridHandler) *HybridDatabaseProvider {
	return &HybridDatabaseProvider{
		originalProvider: provider,
		hybridHandler:    handler,
	}
}

// Database implements sql.DatabaseProvider
func (p *HybridDatabaseProvider) Database(ctx *gmssql.Context, name string) (gmssql.Database, error) {
	// Get the original database
	db, err := p.originalProvider.Database(ctx, name)
	if err != nil {
		return nil, err
	}
	
	// Wrap it with hybrid capabilities
	return &HybridDatabase{
		Database:      db,
		hybridHandler: p.hybridHandler,
	}, nil
}

// HasDatabase implements sql.DatabaseProvider
func (p *HybridDatabaseProvider) HasDatabase(ctx *gmssql.Context, name string) bool {
	return p.originalProvider.HasDatabase(ctx, name)
}

// AllDatabases implements sql.DatabaseProvider
func (p *HybridDatabaseProvider) AllDatabases(ctx *gmssql.Context) []gmssql.Database {
	// Get original databases
	dbs := p.originalProvider.AllDatabases(ctx)
	
	// Wrap them with hybrid capabilities
	hybridDbs := make([]gmssql.Database, len(dbs))
	for i, db := range dbs {
		hybridDbs[i] = &HybridDatabase{
			Database:      db,
			hybridHandler: p.hybridHandler,
		}
	}
	
	return hybridDbs
}

// HybridDatabase wraps a database with hybrid query capabilities
type HybridDatabase struct {
	gmssql.Database
	hybridHandler *HybridHandler
}

// GetTableInsensitive implements sql.Database
func (d *HybridDatabase) GetTableInsensitive(ctx *gmssql.Context, tblName string) (gmssql.Table, bool, error) {
	// Get the original table
	table, exists, err := d.Database.GetTableInsensitive(ctx, tblName)
	if err != nil || !exists {
		return table, exists, err
	}
	
	// Check if this table should use hybrid queries
	if d.hybridHandler.IsTableCached(d.Name(), tblName) {
		// Wrap with hybrid table
		return NewHybridTable(
			table.Name(),
			table.Schema(),
			d.hybridHandler,
			d.Name(),
			table,
		), true, nil
	}
	
	return table, exists, err
}

// GetTableNames implements sql.Database
func (d *HybridDatabase) GetTableNames(ctx *gmssql.Context) ([]string, error) {
	return d.Database.GetTableNames(ctx)
}

// Example usage in main.go
func ExampleIntegration() {
	// This shows how you would integrate the hybrid handler into your main server

	/*
	// In your main.go or server initialization:
	
	// Create hybrid handler
	hybridConfig := hybrid.Config{
		MySQLDSN: "root:password@tcp(remotehost:3306)/production_db",
		LMDBPath: "/var/lib/mysql-server/hybrid_cache",
		Logger:   logger,
	}
	
	hybridHandler, err := hybrid.NewHybridHandler(hybridConfig)
	if err != nil {
		log.Fatalf("Failed to create hybrid handler: %v", err)
	}
	defer hybridHandler.Close()
	
	// Load ACORDO_GM table
	err = hybridHandler.LoadTable("production_db", "ACORDO_GM")
	if err != nil {
		log.Printf("Warning: Failed to load ACORDO_GM: %v", err)
	}
	
	// Create your original database provider
	storage := storage.NewLMDBStorage(lmdbPath, logger)
	originalProvider := provider.NewDatabaseProvider(storage, logger)
	
	// Wrap it with hybrid capabilities
	hybridProvider := hybrid.NewHybridDatabaseProvider(originalProvider, hybridHandler)
	
	// Create engine with hybrid provider
	engine := sqle.NewDefault(hybridProvider)
	
	// Now queries involving ACORDO_GM will automatically use the cached data
	*/
	
	fmt.Println("See comments for integration example")
}

// AdminCommands provides SQL commands to manage the hybrid cache
type AdminCommands struct {
	handler *HybridHandler
}

// NewAdminCommands creates admin commands for hybrid cache management
func NewAdminCommands(handler *HybridHandler) *AdminCommands {
	return &AdminCommands{handler: handler}
}

// ExecuteCommand executes an admin command
func (a *AdminCommands) ExecuteCommand(ctx context.Context, command string, args []string) (string, error) {
	switch command {
	case "HYBRID_LOAD_TABLE":
		if len(args) < 2 {
			return "", fmt.Errorf("usage: HYBRID_LOAD_TABLE <database> <table>")
		}
		err := a.handler.LoadTable(args[0], args[1])
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Table %s.%s loaded into hybrid cache", args[0], args[1]), nil
		
	case "HYBRID_REFRESH_TABLE":
		if len(args) < 2 {
			return "", fmt.Errorf("usage: HYBRID_REFRESH_TABLE <database> <table>")
		}
		err := a.handler.RefreshTable(args[0], args[1])
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Table %s.%s refreshed in hybrid cache", args[0], args[1]), nil
		
	case "HYBRID_STATUS":
		stats := a.handler.GetStats()
		return fmt.Sprintf("Hybrid cache enabled: %v\nCached tables: %v", 
			stats.Enabled, stats.CachedTables), nil
		
	case "HYBRID_ENABLE":
		a.handler.Enable()
		return "Hybrid cache enabled", nil
		
	case "HYBRID_DISABLE":
		a.handler.Disable()
		return "Hybrid cache disabled", nil
		
	default:
		return "", fmt.Errorf("unknown command: %s", command)
	}
}