package storage

import (
	"fmt"
	"strings"

	"github.com/dolthub/go-mysql-server/sql"
)

// FilterOperator represents comparison operations for storage-level filtering
type FilterOperator int

const (
	Equals FilterOperator = iota
	GreaterThan
	LessThan
	GreaterThanOrEqual
	LessThanOrEqual
	NotEquals
	In
	Like
)

// Filter represents a storage-level filter condition
type Filter struct {
	Column   string
	Operator FilterOperator
	Value    interface{}
}

// EnhancedStorage extends the basic Storage interface with advanced features
type EnhancedStorage interface {
	Storage
	
	// Advanced querying with filters pushed down to storage level
	GetRowsWithFilters(database, tableName string, filters []Filter) ([]sql.Row, error)
	
	// Get rows with column projection (only return specified columns)
	GetRowsWithProjection(database, tableName string, columns []string) ([]sql.Row, error)
	
	// Combined filters and projection
	GetRowsWithFiltersAndProjection(database, tableName string, filters []Filter, columns []string) ([]sql.Row, error)
	
	// Statistics for query optimization
	GetTableStats(database, tableName string) (TableStats, error)
	GetColumnStats(database, tableName, columnName string) (ColumnStats, error)
	
	// Index support
	CreateIndex(database, tableName, indexName string, columns []string, unique bool) error
	DropIndex(database, tableName, indexName string) error
	GetIndexes(database, tableName string) ([]IndexInfo, error)
	
	// Bulk operations for better performance
	BulkInsert(database, tableName string, rows []sql.Row) error
	BulkUpdate(database, tableName string, updates []RowUpdate) error
	BulkDelete(database, tableName string, filters []Filter) error
}

// TableStats provides statistics about a table for query optimization
type TableStats struct {
	RowCount     int64
	ApproxSize   int64
	LastModified int64
}

// ColumnStats provides statistics about a column for query optimization
type ColumnStats struct {
	DistinctValues int64
	NullCount      int64
	MinValue       interface{}
	MaxValue       interface{}
	Histogram      []HistogramBucket
}

// HistogramBucket represents a histogram bucket for column value distribution
type HistogramBucket struct {
	LowerBound interface{}
	UpperBound interface{}
	Count      int64
}

// IndexInfo describes an index
type IndexInfo struct {
	Name     string
	Columns  []string
	Unique   bool
	Type     IndexType
}

// IndexType represents different types of indexes
type IndexType int

const (
	BTreeIndex IndexType = iota
	HashIndex
	FullTextIndex
)

// RowUpdate represents a bulk update operation
type RowUpdate struct {
	Filter Filter
	NewValues map[string]interface{}
}

// Enhanced memory storage that implements EnhancedStorage
type EnhancedMemoryStorage struct {
	*MemoryStorage
	indexes map[string]map[string]*Index // database -> table -> index
	stats   map[string]map[string]TableStats // database -> table -> stats
}

// Index represents an in-memory index
type Index struct {
	Name     string
	Columns  []string
	Unique   bool
	Type     IndexType
	Data     map[interface{}][]int // value -> row indices
}

// NewEnhancedMemoryStorage creates an enhanced memory storage
func NewEnhancedMemoryStorage() *EnhancedMemoryStorage {
	return &EnhancedMemoryStorage{
		MemoryStorage: NewMemoryStorage(),
		indexes:       make(map[string]map[string]*Index),
		stats:         make(map[string]map[string]TableStats),
	}
}

// GetRowsWithFilters applies filters at storage level
func (s *EnhancedMemoryStorage) GetRowsWithFilters(database, tableName string, filters []Filter) ([]sql.Row, error) {
	// Get all rows first
	allRows, err := s.GetRows(database, tableName)
	if err != nil {
		return nil, err
	}
	
	if len(filters) == 0 {
		return allRows, nil
	}
	
	// Get table schema to map column names to indices
	schema, err := s.GetTableSchema(database, tableName)
	if err != nil {
		return nil, err
	}
	
	columnMap := make(map[string]int)
	for i, col := range schema {
		columnMap[col.Name] = i
	}
	
	// Apply filters
	var filteredRows []sql.Row
	for _, row := range allRows {
		if s.rowMatchesFilters(row, filters, columnMap) {
			filteredRows = append(filteredRows, row)
		}
	}
	
	return filteredRows, nil
}

// GetRowsWithProjection returns only specified columns
func (s *EnhancedMemoryStorage) GetRowsWithProjection(database, tableName string, columns []string) ([]sql.Row, error) {
	allRows, err := s.GetRows(database, tableName)
	if err != nil {
		return nil, err
	}
	
	if len(columns) == 0 {
		return allRows, nil
	}
	
	// Get column indices
	schema, err := s.GetTableSchema(database, tableName)
	if err != nil {
		return nil, err
	}
	
	columnIndices := make([]int, 0, len(columns))
	for _, requestedCol := range columns {
		for i, schemaCol := range schema {
			if schemaCol.Name == requestedCol {
				columnIndices = append(columnIndices, i)
				break
			}
		}
	}
	
	// Project rows
	var projectedRows []sql.Row
	for _, row := range allRows {
		projectedRow := make(sql.Row, len(columnIndices))
		for i, colIndex := range columnIndices {
			projectedRow[i] = row[colIndex]
		}
		projectedRows = append(projectedRows, projectedRow)
	}
	
	return projectedRows, nil
}

// GetRowsWithFiltersAndProjection combines filtering and projection
func (s *EnhancedMemoryStorage) GetRowsWithFiltersAndProjection(database, tableName string, filters []Filter, columns []string) ([]sql.Row, error) {
	// First apply filters
	filteredRows, err := s.GetRowsWithFilters(database, tableName, filters)
	if err != nil {
		return nil, err
	}
	
	if len(columns) == 0 {
		return filteredRows, nil
	}
	
	// Then apply projection
	schema, err := s.GetTableSchema(database, tableName)
	if err != nil {
		return nil, err
	}
	
	columnIndices := make([]int, 0, len(columns))
	for _, requestedCol := range columns {
		for i, schemaCol := range schema {
			if schemaCol.Name == requestedCol {
				columnIndices = append(columnIndices, i)
				break
			}
		}
	}
	
	var projectedRows []sql.Row
	for _, row := range filteredRows {
		projectedRow := make(sql.Row, len(columnIndices))
		for i, colIndex := range columnIndices {
			projectedRow[i] = row[colIndex]
		}
		projectedRows = append(projectedRows, projectedRow)
	}
	
	return projectedRows, nil
}

// GetTableStats returns statistics about a table
func (s *EnhancedMemoryStorage) GetTableStats(database, tableName string) (TableStats, error) {
	rows, err := s.GetRows(database, tableName)
	if err != nil {
		return TableStats{}, err
	}
	
	return TableStats{
		RowCount:     int64(len(rows)),
		ApproxSize:   int64(len(rows) * 100), // Rough estimate
		LastModified: 0, // Would be actual timestamp in real implementation
	}, nil
}

// GetColumnStats returns statistics about a column
func (s *EnhancedMemoryStorage) GetColumnStats(database, tableName, columnName string) (ColumnStats, error) {
	rows, err := s.GetRows(database, tableName)
	if err != nil {
		return ColumnStats{}, err
	}
	
	schema, err := s.GetTableSchema(database, tableName)
	if err != nil {
		return ColumnStats{}, err
	}
	
	colIndex := -1
	for i, col := range schema {
		if col.Name == columnName {
			colIndex = i
			break
		}
	}
	
	if colIndex == -1 {
		return ColumnStats{}, fmt.Errorf("column %s not found", columnName)
	}
	
	// Calculate statistics
	distinctValues := make(map[interface{}]bool)
	nullCount := int64(0)
	
	for _, row := range rows {
		value := row[colIndex]
		if value == nil {
			nullCount++
		} else {
			distinctValues[value] = true
		}
	}
	
	return ColumnStats{
		DistinctValues: int64(len(distinctValues)),
		NullCount:      nullCount,
		// MinValue, MaxValue, and Histogram would be calculated here
	}, nil
}

// CreateIndex creates an index on specified columns
func (s *EnhancedMemoryStorage) CreateIndex(database, tableName, indexName string, columns []string, unique bool) error {
	// Initialize maps if needed
	if s.indexes[database] == nil {
		s.indexes[database] = make(map[string]*Index)
	}
	
	indexKey := fmt.Sprintf("%s.%s", tableName, indexName)
	if s.indexes[database][indexKey] != nil {
		return fmt.Errorf("index %s already exists", indexName)
	}
	
	// Create index
	index := &Index{
		Name:    indexName,
		Columns: columns,
		Unique:  unique,
		Type:    BTreeIndex,
		Data:    make(map[interface{}][]int),
	}
	
	// Build index data
	rows, err := s.GetRows(database, tableName)
	if err != nil {
		return err
	}
	
	schema, err := s.GetTableSchema(database, tableName)
	if err != nil {
		return err
	}
	
	// Get column indices
	columnIndices := make([]int, len(columns))
	for i, colName := range columns {
		for j, schemaCol := range schema {
			if schemaCol.Name == colName {
				columnIndices[i] = j
				break
			}
		}
	}
	
	// Build index entries
	for rowIndex, row := range rows {
		// Create index key (for multi-column indexes, combine values)
		var indexKey interface{}
		if len(columnIndices) == 1 {
			indexKey = row[columnIndices[0]]
		} else {
			keyParts := make([]interface{}, len(columnIndices))
			for i, colIndex := range columnIndices {
				keyParts[i] = row[colIndex]
			}
			indexKey = fmt.Sprintf("%v", keyParts)
		}
		
		if unique && len(index.Data[indexKey]) > 0 {
			return fmt.Errorf("duplicate key value violates unique constraint")
		}
		
		index.Data[indexKey] = append(index.Data[indexKey], rowIndex)
	}
	
	s.indexes[database][indexKey] = index
	return nil
}

// DropIndex removes an index
func (s *EnhancedMemoryStorage) DropIndex(database, tableName, indexName string) error {
	indexKey := fmt.Sprintf("%s.%s", tableName, indexName)
	
	if s.indexes[database] == nil || s.indexes[database][indexKey] == nil {
		return fmt.Errorf("index %s does not exist", indexName)
	}
	
	delete(s.indexes[database], indexKey)
	return nil
}

// GetIndexes returns information about all indexes on a table
func (s *EnhancedMemoryStorage) GetIndexes(database, tableName string) ([]IndexInfo, error) {
	if s.indexes[database] == nil {
		return []IndexInfo{}, nil
	}
	
	var indexes []IndexInfo
	prefix := tableName + "."
	
	for indexKey, index := range s.indexes[database] {
		if strings.HasPrefix(indexKey, prefix) {
			indexes = append(indexes, IndexInfo{
				Name:    index.Name,
				Columns: index.Columns,
				Unique:  index.Unique,
				Type:    index.Type,
			})
		}
	}
	
	return indexes, nil
}

// BulkInsert performs bulk insert operation
func (s *EnhancedMemoryStorage) BulkInsert(database, tableName string, rows []sql.Row) error {
	for _, row := range rows {
		if err := s.InsertRow(database, tableName, row); err != nil {
			return err
		}
	}
	return nil
}

// BulkUpdate performs bulk update operation
func (s *EnhancedMemoryStorage) BulkUpdate(database, tableName string, updates []RowUpdate) error {
	// Implementation would depend on specific update logic
	return fmt.Errorf("bulk update not implemented in example")
}

// BulkDelete performs bulk delete operation
func (s *EnhancedMemoryStorage) BulkDelete(database, tableName string, filters []Filter) error {
	// Implementation would delete all rows matching filters
	return fmt.Errorf("bulk delete not implemented in example")
}

// rowMatchesFilters checks if a row matches all given filters
func (s *EnhancedMemoryStorage) rowMatchesFilters(row sql.Row, filters []Filter, columnMap map[string]int) bool {
	for _, filter := range filters {
		colIndex, exists := columnMap[filter.Column]
		if !exists {
			continue
		}
		
		rowValue := row[colIndex]
		if !s.valueMatchesFilter(rowValue, filter) {
			return false
		}
	}
	return true
}

// valueMatchesFilter checks if a value matches a specific filter
func (s *EnhancedMemoryStorage) valueMatchesFilter(value interface{}, filter Filter) bool {
	switch filter.Operator {
	case Equals:
		return value == filter.Value
	case GreaterThan:
		return s.compareValues(value, filter.Value) > 0
	case LessThan:
		return s.compareValues(value, filter.Value) < 0
	case GreaterThanOrEqual:
		return s.compareValues(value, filter.Value) >= 0
	case LessThanOrEqual:
		return s.compareValues(value, filter.Value) <= 0
	case NotEquals:
		return value != filter.Value
	case In:
		if values, ok := filter.Value.([]interface{}); ok {
			for _, v := range values {
				if value == v {
					return true
				}
			}
		}
		return false
	default:
		return false
	}
}

// compareValues compares two values (simplified comparison)
func (s *EnhancedMemoryStorage) compareValues(a, b interface{}) int {
	// This is a simplified comparison - real implementation would handle all types
	switch va := a.(type) {
	case int:
		if vb, ok := b.(int); ok {
			if va < vb {
				return -1
			} else if va > vb {
				return 1
			}
			return 0
		}
	case float64:
		if vb, ok := b.(float64); ok {
			if va < vb {
				return -1
			} else if va > vb {
				return 1
			}
			return 0
		}
	case string:
		if vb, ok := b.(string); ok {
			if va < vb {
				return -1
			} else if va > vb {
				return 1
			}
			return 0
		}
	}
	return 0
}

// Ensure we implement the enhanced interface
var _ EnhancedStorage = (*EnhancedMemoryStorage)(nil)