package storage

import (
	"fmt"
	"strings"
	"sync"

	"github.com/dolthub/go-mysql-server/sql"
)

// MemoryStorage implements Storage interface using in-memory storage
type MemoryStorage struct {
	databases map[string]*memoryDatabase
	mu        sync.RWMutex
}

type memoryDatabase struct {
	name   string
	tables map[string]*memoryTable
	mu     sync.RWMutex
}

type memoryTable struct {
	name   string
	schema sql.Schema
	rows   []sql.Row
	mu     sync.RWMutex
}

// NewMemoryStorage creates a new in-memory storage backend
func NewMemoryStorage() *MemoryStorage {
	return &MemoryStorage{
		databases: make(map[string]*memoryDatabase),
	}
}

// Database operations

func (s *MemoryStorage) CreateDatabase(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := s.databases[key]; exists {
		return fmt.Errorf("database %s already exists", name)
	}

	s.databases[key] = &memoryDatabase{
		name:   name,
		tables: make(map[string]*memoryTable),
	}

	return nil
}

func (s *MemoryStorage) DropDatabase(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := strings.ToLower(name)
	if _, exists := s.databases[key]; !exists {
		return fmt.Errorf("database %s does not exist", name)
	}

	delete(s.databases, key)
	return nil
}

func (s *MemoryStorage) HasDatabase(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()

	_, exists := s.databases[strings.ToLower(name)]
	return exists
}

func (s *MemoryStorage) GetDatabaseNames() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()

	names := make([]string, 0, len(s.databases))
	for _, db := range s.databases {
		names = append(names, db.name)
	}
	return names
}

// Table operations

func (s *MemoryStorage) CreateTable(database, tableName string, schema sql.Schema) error {
	db, err := s.getDatabase(database)
	if err != nil {
		return err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	key := strings.ToLower(tableName)
	if _, exists := db.tables[key]; exists {
		return fmt.Errorf("table %s already exists in database %s", tableName, database)
	}

	db.tables[key] = &memoryTable{
		name:   tableName,
		schema: schema,
		rows:   make([]sql.Row, 0),
	}

	return nil
}

func (s *MemoryStorage) DropTable(database, tableName string) error {
	db, err := s.getDatabase(database)
	if err != nil {
		return err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	key := strings.ToLower(tableName)
	if _, exists := db.tables[key]; !exists {
		return fmt.Errorf("table %s does not exist in database %s", tableName, database)
	}

	delete(db.tables, key)
	return nil
}

func (s *MemoryStorage) HasTable(database, tableName string) bool {
	db, err := s.getDatabase(database)
	if err != nil {
		return false
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	_, exists := db.tables[strings.ToLower(tableName)]
	return exists
}

func (s *MemoryStorage) GetTableNames(database string) []string {
	db, err := s.getDatabase(database)
	if err != nil {
		return []string{}
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	names := make([]string, 0, len(db.tables))
	for _, table := range db.tables {
		names = append(names, table.name)
	}
	return names
}

func (s *MemoryStorage) GetTableSchema(database, tableName string) (sql.Schema, error) {
	table, err := s.getTable(database, tableName)
	if err != nil {
		return nil, err
	}

	table.mu.RLock()
	defer table.mu.RUnlock()

	return table.schema, nil
}

// Row operations

func (s *MemoryStorage) InsertRow(database, tableName string, row sql.Row) error {
	table, err := s.getTable(database, tableName)
	if err != nil {
		return err
	}

	table.mu.Lock()
	defer table.mu.Unlock()

	table.rows = append(table.rows, row)
	return nil
}

func (s *MemoryStorage) UpdateRow(database, tableName string, oldRow, newRow sql.Row) error {
	table, err := s.getTable(database, tableName)
	if err != nil {
		return err
	}

	table.mu.Lock()
	defer table.mu.Unlock()

	// Find the old row and replace it
	for i, row := range table.rows {
		if s.rowsEqual(row, oldRow) {
			table.rows[i] = newRow
			return nil
		}
	}

	return fmt.Errorf("row not found for update")
}

func (s *MemoryStorage) DeleteRow(database, tableName string, row sql.Row) error {
	table, err := s.getTable(database, tableName)
	if err != nil {
		return err
	}

	table.mu.Lock()
	defer table.mu.Unlock()

	// Find and remove the row
	for i, tableRow := range table.rows {
		if s.rowsEqual(tableRow, row) {
			table.rows = append(table.rows[:i], table.rows[i+1:]...)
			return nil
		}
	}

	return fmt.Errorf("row not found for deletion")
}

func (s *MemoryStorage) GetRows(database, tableName string) ([]sql.Row, error) {
	table, err := s.getTable(database, tableName)
	if err != nil {
		return nil, err
	}

	table.mu.RLock()
	defer table.mu.RUnlock()

	// Return a copy of the rows
	result := make([]sql.Row, len(table.rows))
	copy(result, table.rows)
	return result, nil
}

// Helper methods

func (s *MemoryStorage) getDatabase(name string) (*memoryDatabase, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	db, exists := s.databases[strings.ToLower(name)]
	if !exists {
		// Auto-create database if it doesn't exist
		s.mu.RUnlock()
		err := s.CreateDatabase(name)
		s.mu.RLock()
		if err != nil {
			return nil, err
		}
		db = s.databases[strings.ToLower(name)]
	}
	return db, nil
}

func (s *MemoryStorage) getTable(database, tableName string) (*memoryTable, error) {
	db, err := s.getDatabase(database)
	if err != nil {
		return nil, err
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	table, exists := db.tables[strings.ToLower(tableName)]
	if !exists {
		return nil, fmt.Errorf("table %s does not exist in database %s", tableName, database)
	}

	return table, nil
}

func (s *MemoryStorage) rowsEqual(row1, row2 sql.Row) bool {
	if len(row1) != len(row2) {
		return false
	}

	for i, val1 := range row1 {
		val2 := row2[i]
		if val1 != val2 {
			return false
		}
	}

	return true
}

// Ensure we implement the Storage interface
var _ Storage = (*MemoryStorage)(nil)