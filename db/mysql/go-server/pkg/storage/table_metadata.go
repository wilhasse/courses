package storage

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// TableStatistics contains detailed statistics about a table
type TableStatistics struct {
	TableMetadata
	
	// Query patterns
	SelectCount      int64     // Number of SELECT queries
	InsertCount      int64     // Number of INSERT queries
	UpdateCount      int64     // Number of UPDATE queries
	DeleteCount      int64     // Number of DELETE queries
	JoinCount        int64     // Number of queries involving JOINs
	AggregationCount int64     // Number of queries with GROUP BY/aggregations
	
	// Performance metrics
	AvgQueryTimeMs   float64   // Average query execution time
	LastQueryTimeMs  float64   // Last query execution time
	
	// Data characteristics
	Cardinality      map[string]int64 // Column name -> distinct values
	NullPercentage   map[string]float64 // Column name -> null percentage
	
	// Storage metrics
	CompressionRatio float64   // For chDB tables
	LastOptimized    time.Time // Last OPTIMIZE TABLE execution
	
	// Access patterns
	HourlyAccess     [24]int64 // Access count by hour of day
	DailyAccess      [7]int64  // Access count by day of week
	
	// Update tracking
	LastModified     time.Time
	ModificationRate float64   // Updates per hour
}

// MetadataStore manages table metadata persistence
type MetadataStore struct {
	storage  Storage
	cache    map[string]map[string]*TableStatistics // database -> table -> stats
	mutex    sync.RWMutex
	
	// Background tasks
	persistTicker *time.Ticker
	updateTicker  *time.Ticker
	done          chan bool
}

// NewMetadataStore creates a new metadata store
func NewMetadataStore(storage Storage) *MetadataStore {
	store := &MetadataStore{
		storage: storage,
		cache:   make(map[string]map[string]*TableStatistics),
		done:    make(chan bool),
	}
	
	// Load existing metadata
	store.loadMetadata()
	
	// Start background tasks
	store.startBackgroundTasks()
	
	return store
}

// Close stops background tasks and persists metadata
func (m *MetadataStore) Close() error {
	// Stop background tasks
	close(m.done)
	if m.persistTicker != nil {
		m.persistTicker.Stop()
	}
	if m.updateTicker != nil {
		m.updateTicker.Stop()
	}
	
	// Final persist
	return m.persistMetadata()
}

// startBackgroundTasks starts periodic metadata operations
func (m *MetadataStore) startBackgroundTasks() {
	// Persist metadata every 5 minutes
	m.persistTicker = time.NewTicker(5 * time.Minute)
	go func() {
		for {
			select {
			case <-m.persistTicker.C:
				m.persistMetadata()
			case <-m.done:
				return
			}
		}
	}()
	
	// Update statistics every minute
	m.updateTicker = time.NewTicker(1 * time.Minute)
	go func() {
		for {
			select {
			case <-m.updateTicker.C:
				m.updateStatistics()
			case <-m.done:
				return
			}
		}
	}()
}

// GetTableStats returns statistics for a table
func (m *MetadataStore) GetTableStats(database, table string) *TableStatistics {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	if dbStats, ok := m.cache[database]; ok {
		if tableStats, ok := dbStats[table]; ok {
			return tableStats
		}
	}
	
	// Create new stats if not exists
	return m.createTableStats(database, table)
}

// createTableStats creates initial statistics for a table
func (m *MetadataStore) createTableStats(database, table string) *TableStatistics {
	now := time.Now()
	stats := &TableStatistics{
		TableMetadata: TableMetadata{
			LastAccessed:   now,
			StorageBackend: "lmdb", // Default
		},
		Cardinality:    make(map[string]int64),
		NullPercentage: make(map[string]float64),
		LastModified:   now,
	}
	
	m.mutex.Lock()
	if m.cache[database] == nil {
		m.cache[database] = make(map[string]*TableStatistics)
	}
	m.cache[database][table] = stats
	m.mutex.Unlock()
	
	return stats
}

// RecordQuery records a query execution for statistics
func (m *MetadataStore) RecordQuery(database, table, queryType string, executionTimeMs float64) {
	stats := m.GetTableStats(database, table)
	
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	// Update access time and count
	stats.LastAccessed = time.Now()
	stats.AccessCount++
	
	// Update query type counts
	switch queryType {
	case "SELECT":
		stats.SelectCount++
	case "INSERT":
		stats.InsertCount++
		stats.LastModified = time.Now()
	case "UPDATE":
		stats.UpdateCount++
		stats.LastModified = time.Now()
	case "DELETE":
		stats.DeleteCount++
		stats.LastModified = time.Now()
	}
	
	// Update performance metrics
	stats.LastQueryTimeMs = executionTimeMs
	if stats.AvgQueryTimeMs == 0 {
		stats.AvgQueryTimeMs = executionTimeMs
	} else {
		// Rolling average
		stats.AvgQueryTimeMs = (stats.AvgQueryTimeMs*float64(stats.AccessCount-1) + executionTimeMs) / float64(stats.AccessCount)
	}
	
	// Update hourly and daily patterns
	hour := time.Now().Hour()
	stats.HourlyAccess[hour]++
	
	weekday := int(time.Now().Weekday())
	stats.DailyAccess[weekday]++
}

// RecordJoin records that a table was used in a JOIN
func (m *MetadataStore) RecordJoin(database, table string) {
	stats := m.GetTableStats(database, table)
	
	m.mutex.Lock()
	stats.JoinCount++
	m.mutex.Unlock()
}

// RecordAggregation records that a table was used with aggregation
func (m *MetadataStore) RecordAggregation(database, table string) {
	stats := m.GetTableStats(database, table)
	
	m.mutex.Lock()
	stats.AggregationCount++
	m.mutex.Unlock()
}

// AnalyzeTableCharacteristics analyzes a table to determine best storage
func (m *MetadataStore) AnalyzeTableCharacteristics(database, table string) string {
	stats := m.GetTableStats(database, table)
	
	// Calculate scores for each storage backend
	lmdbScore := 0.0
	chdbScore := 0.0
	
	// Size factor
	if stats.RowCount < 1_000_000 {
		lmdbScore += 10
	} else if stats.RowCount > 10_000_000 {
		chdbScore += 10
	} else {
		// Medium size, check other factors
		lmdbScore += 5
		chdbScore += 5
	}
	
	// Access pattern factor
	accessRatio := float64(stats.AccessCount) / (float64(stats.RowCount) + 1)
	if accessRatio > 0.1 { // Frequently accessed
		lmdbScore += 8
	} else {
		chdbScore += 3
	}
	
	// Query pattern factor
	totalQueries := stats.SelectCount + stats.InsertCount + stats.UpdateCount + stats.DeleteCount
	if totalQueries > 0 {
		// High percentage of analytical queries
		analyticalRatio := float64(stats.AggregationCount+stats.JoinCount) / float64(totalQueries)
		chdbScore += analyticalRatio * 10
		
		// High percentage of modifications
		modificationRatio := float64(stats.InsertCount+stats.UpdateCount+stats.DeleteCount) / float64(totalQueries)
		lmdbScore += modificationRatio * 8
	}
	
	// Performance factor
	if stats.AvgQueryTimeMs > 100 { // Slow queries might benefit from chDB
		chdbScore += 5
	}
	
	// Make decision
	if chdbScore > lmdbScore {
		return "chdb"
	}
	return "lmdb"
}

// GetMigrationCandidates returns tables that should be migrated to different storage
func (m *MetadataStore) GetMigrationCandidates() []MigrationCandidate {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	var candidates []MigrationCandidate
	
	for database, dbStats := range m.cache {
		for table, stats := range dbStats {
			recommendedBackend := m.AnalyzeTableCharacteristics(database, table)
			
			if recommendedBackend != stats.StorageBackend {
				candidate := MigrationCandidate{
					Database:        database,
					Table:           table,
					CurrentBackend:  stats.StorageBackend,
					RecommendedBackend: recommendedBackend,
					RowCount:        stats.RowCount,
					Reason:          m.getMigrationReason(stats, recommendedBackend),
				}
				candidates = append(candidates, candidate)
			}
		}
	}
	
	return candidates
}

// MigrationCandidate represents a table that should be migrated
type MigrationCandidate struct {
	Database           string
	Table              string
	CurrentBackend     string
	RecommendedBackend string
	RowCount           int64
	Reason             string
}

// getMigrationReason explains why a table should be migrated
func (m *MetadataStore) getMigrationReason(stats *TableStatistics, recommended string) string {
	if recommended == "chdb" {
		if stats.RowCount > 10_000_000 {
			return fmt.Sprintf("Large table with %d rows", stats.RowCount)
		}
		if stats.AggregationCount > stats.SelectCount/2 {
			return "High percentage of analytical queries"
		}
		if stats.JoinCount > stats.SelectCount/3 {
			return "Frequently used in JOINs"
		}
		return "Table characteristics favor analytical storage"
	}
	
	// Recommended LMDB
	if stats.RowCount < 1_000_000 {
		return fmt.Sprintf("Small table with %d rows", stats.RowCount)
	}
	modRate := float64(stats.UpdateCount+stats.InsertCount+stats.DeleteCount) / float64(stats.SelectCount+1)
	if modRate > 0.3 {
		return "High modification rate"
	}
	return "Table characteristics favor transactional storage"
}

// persistMetadata saves metadata to storage
func (m *MetadataStore) persistMetadata() error {
	m.mutex.RLock()
	data, err := json.Marshal(m.cache)
	m.mutex.RUnlock()
	
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}
	
	// Store in system database
	// This is a simplified version - in production you'd want proper schema
	// For now, we'll just log that we would persist
	// TODO: Implement actual persistence to LMDB system tables
	_ = data // Will be used when persistence is implemented
	
	return nil
}

// loadMetadata loads metadata from storage
func (m *MetadataStore) loadMetadata() error {
	// TODO: Implement actual loading from LMDB system tables
	// For now, start with empty cache
	return nil
}

// updateStatistics updates derived statistics
func (m *MetadataStore) updateStatistics() {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	now := time.Now()
	
	for _, dbStats := range m.cache {
		for _, stats := range dbStats {
			// Calculate modification rate (updates per hour)
			if stats.LastModified.After(time.Time{}) {
				hoursSinceModified := now.Sub(stats.LastModified).Hours()
				if hoursSinceModified > 0 {
					totalMods := stats.InsertCount + stats.UpdateCount + stats.DeleteCount
					stats.ModificationRate = float64(totalMods) / hoursSinceModified
				}
			}
		}
	}
}

// GetDatabaseSummary returns a summary of all tables in a database
func (m *MetadataStore) GetDatabaseSummary(database string) DatabaseSummary {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	
	summary := DatabaseSummary{
		Database: database,
		Tables:   make([]TableSummary, 0),
	}
	
	if dbStats, ok := m.cache[database]; ok {
		for table, stats := range dbStats {
			tableSummary := TableSummary{
				Table:           table,
				RowCount:        stats.RowCount,
				SizeBytes:       stats.SizeBytes,
				StorageBackend:  stats.StorageBackend,
				AccessCount:     stats.AccessCount,
				LastAccessed:    stats.LastAccessed,
				IsAnalytical:    stats.IsAnalytical,
				AvgQueryTimeMs:  stats.AvgQueryTimeMs,
			}
			summary.Tables = append(summary.Tables, tableSummary)
			summary.TotalRows += stats.RowCount
			summary.TotalSizeBytes += stats.SizeBytes
		}
	}
	
	return summary
}

// DatabaseSummary contains summary statistics for a database
type DatabaseSummary struct {
	Database       string
	Tables         []TableSummary
	TotalRows      int64
	TotalSizeBytes int64
}

// TableSummary contains summary statistics for a table
type TableSummary struct {
	Table          string
	RowCount       int64
	SizeBytes      int64
	StorageBackend string
	AccessCount    int64
	LastAccessed   time.Time
	IsAnalytical   bool
	AvgQueryTimeMs float64
}