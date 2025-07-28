package storage

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/types"
	"github.com/dolthub/vitess/go/sqltypes"
	"github.com/rs/zerolog"
	"wellquite.org/golmdb"
)

// LMDBStorage implements the Storage interface using LMDB
type LMDBStorage struct {
	client *golmdb.LMDBClient
	logger zerolog.Logger
	mu     sync.RWMutex
}

// SerializableColumn represents a column that can be JSON serialized
type SerializableColumn struct {
	Name         string `json:"name"`
	TypeName     string `json:"type_name"`
	Nullable     bool   `json:"nullable"`
	PrimaryKey   bool   `json:"primary_key"`
	AutoIncrement bool  `json:"auto_increment"`
	Default      string `json:"default,omitempty"`
}

// SerializableSchema represents a schema that can be JSON serialized
type SerializableSchema struct {
	Columns []SerializableColumn `json:"columns"`
}

// NewLMDBStorage creates a new LMDB storage instance
func NewLMDBStorage(dbPath string, logger zerolog.Logger) (*LMDBStorage, error) {
	client, err := golmdb.NewLMDB(
		logger,
		dbPath,
		0644,
		126,  // max readers
		50,   // max DBs (we need one per database)
		0,    // environment flags
		1000, // batch size
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create LMDB client: %w", err)
	}

	return &LMDBStorage{
		client: client,
		logger: logger,
	}, nil
}

// Close closes the LMDB connection
func (s *LMDBStorage) Close() error {
	s.client.TerminateSync()
	return nil
}

// schemaToSerializable converts sql.Schema to SerializableSchema
func schemaToSerializable(schema sql.Schema) SerializableSchema {
	serializable := SerializableSchema{
		Columns: make([]SerializableColumn, len(schema)),
	}
	
	for i, col := range schema {
		serializable.Columns[i] = SerializableColumn{
			Name:         col.Name,
			TypeName:     col.Type.String(),
			Nullable:     col.Nullable,
			PrimaryKey:   col.PrimaryKey,
			AutoIncrement: col.AutoIncrement,
		}
		if col.Default != nil {
			serializable.Columns[i].Default = col.Default.String()
		}
	}
	
	return serializable
}

// serializableToSchema converts SerializableSchema to sql.Schema
func serializableToSchema(serializable SerializableSchema) sql.Schema {
	schema := make(sql.Schema, len(serializable.Columns))
	
	for i, col := range serializable.Columns {
		sqlType := parseTypeFromString(col.TypeName)
		schema[i] = &sql.Column{
			Name:         col.Name,
			Type:         sqlType,
			Nullable:     col.Nullable,
			PrimaryKey:   col.PrimaryKey,
			AutoIncrement: col.AutoIncrement,
		}
		// Note: Default value parsing is complex and often not needed for basic operations
	}
	
	return schema
}

// parseTypeFromString converts a type string back to sql.Type
func parseTypeFromString(typeStr string) sql.Type {
	// Handle common types used in our application
	switch {
	case strings.HasPrefix(typeStr, "INT"):
		return types.Int32
	case strings.HasPrefix(typeStr, "VARCHAR"):
		// Extract length if present, default to 255
		if strings.Contains(typeStr, "(") && strings.Contains(typeStr, ")") {
			start := strings.Index(typeStr, "(") + 1
			end := strings.Index(typeStr, ")")
			if lengthStr := typeStr[start:end]; lengthStr != "" {
				if length, err := strconv.Atoi(lengthStr); err == nil {
					return types.MustCreateStringWithDefaults(sqltypes.VarChar, int64(length))
				}
			}
		}
		return types.MustCreateStringWithDefaults(sqltypes.VarChar, 255)
	case strings.HasPrefix(typeStr, "TEXT"):
		return types.Text
	case strings.HasPrefix(typeStr, "DECIMAL"):
		return types.Float64
	case strings.HasPrefix(typeStr, "TIMESTAMP"):
		return types.Timestamp
	case strings.HasPrefix(typeStr, "DATETIME"):
		return types.Datetime
	case strings.HasPrefix(typeStr, "DATE"):
		return types.Date
	case strings.HasPrefix(typeStr, "FLOAT"), strings.HasPrefix(typeStr, "DOUBLE"):
		return types.Float64
	default:
		// Fallback to TEXT for unknown types
		return types.Text
	}
}

// LMDB Key Design:
// databases -> ["testdb", "mydb"]
// db:testdb:tables -> ["users", "products"] 
// db:testdb:table:users:schema -> json schema
// db:testdb:table:users:nextid -> auto increment counter
// db:testdb:table:users:row:1 -> json row data

// Database operations

func (s *LMDBStorage) CreateDatabase(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("main", golmdb.Create)
		if err != nil {
			return err
		}

		// Check if database already exists
		databases, _ := s.getDatabaseListTxn(txn, db)
		for _, dbName := range databases {
			if dbName == name {
				// Database already exists, return nil for CREATE IF NOT EXISTS
				return nil
			}
		}

		// Add to database list
		databases = append(databases, name)
		data, _ := json.Marshal(databases)
		err = txn.Put(db, []byte("databases"), data, 0)
		if err != nil {
			return err
		}

		// Initialize empty table list for this database
		key := fmt.Sprintf("db:%s:tables", name)
		emptyTables := []string{}
		tableData, _ := json.Marshal(emptyTables)
		return txn.Put(db, []byte(key), tableData, 0)
	})
}

func (s *LMDBStorage) DropDatabase(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("main", golmdb.Create)
		if err != nil {
			return err
		}

		// Get current database list
		databases, err := s.getDatabaseListTxn(txn, db)
		if err != nil {
			return err
		}

		// Remove database from list
		newDatabases := []string{}
		found := false
		for _, dbName := range databases {
			if dbName != name {
				newDatabases = append(newDatabases, dbName)
			} else {
				found = true
			}
		}

		if !found {
			return fmt.Errorf("database %s does not exist", name)
		}

		// Update database list
		data, _ := json.Marshal(newDatabases)
		err = txn.Put(db, []byte("databases"), data, 0)
		if err != nil {
			return err
		}

		// Delete all database keys (tables, schemas, rows)
		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		prefix := fmt.Sprintf("db:%s:", name)
		keysToDelete := []string{}

		// Collect keys to delete
		key, _, err := cursor.First()
		for err == nil {
			keyStr := string(key)
			if strings.HasPrefix(keyStr, prefix) {
				keysToDelete = append(keysToDelete, keyStr)
			}
			key, _, err = cursor.Next()
		}

		// Delete collected keys
		for _, key := range keysToDelete {
			err = txn.Delete(db, []byte(key), nil)
			if err != nil {
				return err
			}
		}

		return nil
	})
}

func (s *LMDBStorage) HasDatabase(name string) bool {
	var exists bool
	s.client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("main", 0)
		if err != nil {
			return nil
		}

		databases, _ := s.getDatabaseListTxn(txn, db)
		for _, dbName := range databases {
			if dbName == name {
				exists = true
				break
			}
		}
		return nil
	})
	return exists
}

func (s *LMDBStorage) GetDatabaseNames() []string {
	var databases []string
	s.client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("main", 0)
		if err != nil {
			return nil
		}
		databases, _ = s.getDatabaseListTxn(txn, db)
		return nil
	})
	return databases
}

func (s *LMDBStorage) getDatabaseListTxn(txn interface{}, db golmdb.DBRef) ([]string, error) {
	var data []byte
	var err error
	
	switch t := txn.(type) {
	case *golmdb.ReadOnlyTxn:
		data, err = t.Get(db, []byte("databases"))
	case *golmdb.ReadWriteTxn:
		data, err = t.Get(db, []byte("databases"))
	default:
		return nil, fmt.Errorf("unknown transaction type")
	}
	
	if err != nil {
		return []string{}, nil // No databases yet
	}

	var databases []string
	json.Unmarshal(data, &databases)
	return databases, nil
}

// Table operations

func (s *LMDBStorage) CreateTable(database, tableName string, schema sql.Schema) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("main", golmdb.Create)
		if err != nil {
			return err
		}

		// Get current table list
		tableKey := fmt.Sprintf("db:%s:tables", database)
		data, err := txn.Get(db, []byte(tableKey))
		
		var tables []string
		if err == nil {
			json.Unmarshal(data, &tables)
		}

		// Check if table already exists
		for _, tbl := range tables {
			if tbl == tableName {
				return fmt.Errorf("table %s already exists", tableName)
			}
		}

		// Add table to list
		tables = append(tables, tableName)
		tableData, _ := json.Marshal(tables)
		err = txn.Put(db, []byte(tableKey), tableData, 0)
		if err != nil {
			return err
		}

		// Store table schema
		schemaKey := fmt.Sprintf("db:%s:table:%s:schema", database, tableName)
		serializableSchema := schemaToSerializable(schema)
		schemaData, _ := json.Marshal(serializableSchema)
		err = txn.Put(db, []byte(schemaKey), schemaData, 0)
		if err != nil {
			return err
		}

		// Initialize auto increment counter
		counterKey := fmt.Sprintf("db:%s:table:%s:nextid", database, tableName)
		return txn.Put(db, []byte(counterKey), []byte("1"), 0)
	})
}

func (s *LMDBStorage) DropTable(database, tableName string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("main", golmdb.Create)
		if err != nil {
			return err
		}

		// Get current table list
		tableKey := fmt.Sprintf("db:%s:tables", database)
		data, err := txn.Get(db, []byte(tableKey))
		if err != nil {
			return fmt.Errorf("database %s does not exist", database)
		}

		var tables []string
		json.Unmarshal(data, &tables)

		// Remove table from list
		newTables := []string{}
		found := false
		for _, tbl := range tables {
			if tbl != tableName {
				newTables = append(newTables, tbl)
			} else {
				found = true
			}
		}

		if !found {
			return fmt.Errorf("table %s does not exist", tableName)
		}

		// Update table list
		tableData, _ := json.Marshal(newTables)
		err = txn.Put(db, []byte(tableKey), tableData, 0)
		if err != nil {
			return err
		}

		// Delete table schema and all rows
		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		prefix := fmt.Sprintf("db:%s:table:%s:", database, tableName)
		keysToDelete := []string{}

		// Collect keys to delete
		key, _, err := cursor.First()
		for err == nil {
			keyStr := string(key)
			if strings.HasPrefix(keyStr, prefix) {
				keysToDelete = append(keysToDelete, keyStr)
			}
			key, _, err = cursor.Next()
		}

		// Delete collected keys
		for _, key := range keysToDelete {
			err = txn.Delete(db, []byte(key), nil)
			if err != nil {
				return err
			}
		}

		return nil
	})
}

func (s *LMDBStorage) HasTable(database, tableName string) bool {
	tables := s.GetTableNames(database)
	for _, table := range tables {
		if table == tableName {
			return true
		}
	}
	return false
}

func (s *LMDBStorage) GetTableNames(database string) []string {
	var tables []string
	s.client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("main", 0)
		if err != nil {
			return nil // Return empty list on error
		}

		tableKey := fmt.Sprintf("db:%s:tables", database)
		data, err := txn.Get(db, []byte(tableKey))
		if err != nil {
			return nil // Return empty list if database doesn't exist
		}

		json.Unmarshal(data, &tables)
		return nil
	})
	return tables
}

func (s *LMDBStorage) GetTableSchema(database, tableName string) (sql.Schema, error) {
	var schema sql.Schema
	err := s.client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("main", 0)
		if err != nil {
			return err
		}

		schemaKey := fmt.Sprintf("db:%s:table:%s:schema", database, tableName)
		data, err := txn.Get(db, []byte(schemaKey))
		if err != nil {
			return fmt.Errorf("table %s does not exist", tableName)
		}

		var serializableSchema SerializableSchema
		err = json.Unmarshal(data, &serializableSchema)
		if err != nil {
			return err
		}
		schema = serializableToSchema(serializableSchema)
		return nil
	})
	return schema, err
}

// Row operations

func (s *LMDBStorage) InsertRow(database, tableName string, row sql.Row) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("main", golmdb.Create)
		if err != nil {
			return err
		}

		// Get next ID for auto increment
		counterKey := fmt.Sprintf("db:%s:table:%s:nextid", database, tableName)
		counterData, err := txn.Get(db, []byte(counterKey))
		if err != nil {
			return fmt.Errorf("table %s does not exist", tableName)
		}

		nextID, _ := strconv.Atoi(string(counterData))
		
		// Store row
		rowKey := fmt.Sprintf("db:%s:table:%s:row:%d", database, tableName, nextID)
		rowData, _ := json.Marshal(row)
		err = txn.Put(db, []byte(rowKey), rowData, 0)
		if err != nil {
			return err
		}

		// Update counter
		newCounter := strconv.Itoa(nextID + 1)
		return txn.Put(db, []byte(counterKey), []byte(newCounter), 0)
	})
}

func (s *LMDBStorage) UpdateRow(database, tableName string, oldRow, newRow sql.Row) error {
	// For simplicity, we'll implement this as delete + insert
	// In a real implementation, you'd want to find by primary key
	return fmt.Errorf("UpdateRow not implemented for LMDB storage")
}

func (s *LMDBStorage) DeleteRow(database, tableName string, row sql.Row) error {
	// For simplicity, not implemented
	return fmt.Errorf("DeleteRow not implemented for LMDB storage")
}

func (s *LMDBStorage) GetRows(database, tableName string) ([]sql.Row, error) {
	var rows []sql.Row
	err := s.client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("main", 0)
		if err != nil {
			return err
		}

		// Use cursor to iterate through all rows
		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		prefix := fmt.Sprintf("db:%s:table:%s:row:", database, tableName)
		
		key, data, err := cursor.First()
		for err == nil {
			keyStr := string(key)
			if strings.HasPrefix(keyStr, prefix) {
				var row sql.Row
				json.Unmarshal(data, &row)
				rows = append(rows, row)
			}
			key, data, err = cursor.Next()
		}

		return nil
	})
	return rows, err
}