package hybrid

import (
	"database/sql"
	"fmt"
	"log"
	"sync"

	gmssql "github.com/dolthub/go-mysql-server/sql"
	"github.com/rs/zerolog"
	"wellquite.org/golmdb"
)

// HybridHandler manages hybrid queries between MySQL and LMDB
type HybridHandler struct {
	dataLoader     *DataLoader
	SQLParser      *SQLParser      // Exported for external use
	QueryRewriter  *QueryRewriter  // Exported for external use
	joinExecutor   *JoinExecutor
	logger         zerolog.Logger
	mu             sync.RWMutex
	enabled        bool
}

// Config contains configuration for the hybrid handler
type Config struct {
	MySQLDSN  string
	LMDBPath  string
	Logger    zerolog.Logger
}

// NewHybridHandler creates a new hybrid query handler
func NewHybridHandler(config Config) (*HybridHandler, error) {
	// Create data loader
	dataLoader, err := NewDataLoader(config.MySQLDSN, config.LMDBPath, config.Logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create data loader: %w", err)
	}

	// Create SQL parser
	sqlParser := NewSQLParser()

	// Create query rewriter
	queryRewriter := NewQueryRewriter(sqlParser)

	// Get MySQL connection and LMDB client from data loader
	// Note: In a real implementation, you'd expose these from DataLoader
	mysqlConn, err := sql.Open("mysql", config.MySQLDSN)
	if err != nil {
		dataLoader.Close()
		return nil, fmt.Errorf("failed to open MySQL connection: %w", err)
	}

	lmdbClient, err := golmdb.NewLMDB(
		config.Logger,
		config.LMDBPath,
		0644,
		126,
		50,
		0,
		1000,
	)
	if err != nil {
		mysqlConn.Close()
		dataLoader.Close()
		return nil, fmt.Errorf("failed to create LMDB client: %w", err)
	}

	// Create join executor
	joinExecutor := NewJoinExecutor(mysqlConn, lmdbClient)

	return &HybridHandler{
		dataLoader:     dataLoader,
		SQLParser:      sqlParser,
		QueryRewriter:  queryRewriter,
		joinExecutor:   joinExecutor,
		logger:         config.Logger,
		enabled:        true,
	}, nil
}

// Close closes all resources
func (h *HybridHandler) Close() error {
	return h.dataLoader.Close()
}

// Enable enables hybrid query handling
func (h *HybridHandler) Enable() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.enabled = true
	log.Println("Hybrid query handling enabled")
}

// Disable disables hybrid query handling
func (h *HybridHandler) Disable() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.enabled = false
	log.Println("Hybrid query handling disabled")
}

// IsEnabled returns whether hybrid query handling is enabled
func (h *HybridHandler) IsEnabled() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.enabled
}

// LoadTable loads a table from MySQL to LMDB cache
func (h *HybridHandler) LoadTable(database, table string) error {
	log.Printf("Loading table %s.%s to LMDB cache", database, table)
	
	// Load the table data
	if err := h.dataLoader.LoadTableToLMDB(database, table); err != nil {
		return fmt.Errorf("failed to load table: %w", err)
	}

	// Register the table as cached in the parser
	h.SQLParser.RegisterCachedTable(database, table)
	
	log.Printf("Successfully loaded table %s.%s to cache", database, table)
	return nil
}

// RefreshTable refreshes a cached table from MySQL
func (h *HybridHandler) RefreshTable(database, table string) error {
	log.Printf("Refreshing cached table %s.%s", database, table)
	return h.dataLoader.RefreshTable(database, table)
}

// IsTableCached checks if a table is cached
func (h *HybridHandler) IsTableCached(database, table string) bool {
	return h.dataLoader.IsTableCached(database, table)
}

// ExecuteQuery executes a query using hybrid approach if applicable
func (h *HybridHandler) ExecuteQuery(query string, currentDatabase string) (*QueryResult, error) {
	if !h.IsEnabled() {
		return nil, fmt.Errorf("hybrid query handling is disabled")
	}

	log.Printf("Executing hybrid query: %s", query)

	// Analyze the query
	analysis, err := h.SQLParser.AnalyzeQuery(query, currentDatabase)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query: %w", err)
	}

	// If no cached tables are involved, return nil to use normal execution
	if !analysis.HasCachedTable {
		log.Println("No cached tables in query, using normal execution")
		return nil, nil
	}

	// Rewrite the query if needed
	rewriteResult, err := h.QueryRewriter.RewriteQuery(query, currentDatabase)
	if err != nil {
		return nil, fmt.Errorf("failed to rewrite query: %w", err)
	}

	// Execute the hybrid query
	result, err := h.joinExecutor.ExecuteHybridQuery(rewriteResult, currentDatabase)
	if err != nil {
		return nil, fmt.Errorf("failed to execute hybrid query: %w", err)
	}

	log.Printf("Hybrid query returned %d rows", len(result.Rows))
	return result, nil
}

// ConvertToGMSRows converts QueryResult to go-mysql-server rows
func (h *HybridHandler) ConvertToGMSRows(result *QueryResult, schema gmssql.Schema) ([]gmssql.Row, error) {
	if result == nil {
		return nil, nil
	}

	// Create column index map
	colIndex := make(map[string]int)
	for i, col := range result.Columns {
		colIndex[col] = i
	}

	// Convert rows
	gmsRows := make([]gmssql.Row, len(result.Rows))
	for i, row := range result.Rows {
		gmsRow := make(gmssql.Row, len(schema))
		
		// Map columns from result to schema order
		for j, col := range schema {
			if idx, ok := colIndex[col.Name]; ok && idx < len(row) {
				gmsRow[j] = row[idx]
			} else {
				// Column not found in result, use null
				gmsRow[j] = nil
			}
		}
		
		gmsRows[i] = gmsRow
	}

	return gmsRows, nil
}

// GetCachedTableData retrieves all data from a cached table
func (h *HybridHandler) GetCachedTableData(database, table string) ([]gmssql.Row, gmssql.Schema, error) {
	if !h.IsTableCached(database, table) {
		return nil, nil, fmt.Errorf("table %s.%s is not cached", database, table)
	}

	// Get the data using join executor's method
	result, err := h.joinExecutor.getCachedTableData(database, table)
	if err != nil {
		return nil, nil, err
	}

	// Get schema from data loader (we need to expose this method)
	schema, err := h.getTableSchema(database, table)
	if err != nil {
		return nil, nil, err
	}

	// Convert to GMS rows
	rows, err := h.ConvertToGMSRows(result, schema)
	if err != nil {
		return nil, nil, err
	}

	return rows, schema, nil
}

// getTableSchema retrieves the schema for a table
func (h *HybridHandler) getTableSchema(database, table string) (gmssql.Schema, error) {
	// This is a simplified version - in production you'd get this from LMDB
	// For now, we'll use the data loader's method
	return h.dataLoader.getTableSchema(database, table)
}

// Stats returns statistics about the hybrid handler
type Stats struct {
	Enabled      bool
	CachedTables []string
}

// GetStats returns current statistics
func (h *HybridHandler) GetStats() Stats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	stats := Stats{
		Enabled:      h.enabled,
		CachedTables: make([]string, 0),
	}

	// Get list of cached tables
	// In a real implementation, you'd iterate through LMDB to get this
	// For now, we'll just return ACORDO_GM if it's registered
	if h.SQLParser.IsCachedTable("", "ACORDO_GM") {
		stats.CachedTables = append(stats.CachedTables, "ACORDO_GM")
	}
	if h.SQLParser.IsCachedTable("", "employees") {
		stats.CachedTables = append(stats.CachedTables, "employees")
	}

	return stats
}