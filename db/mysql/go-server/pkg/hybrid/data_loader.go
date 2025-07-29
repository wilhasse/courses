package hybrid

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	gmssql "github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/types"
	"github.com/dolthub/vitess/go/sqltypes"
	_ "github.com/go-sql-driver/mysql"
	"github.com/rs/zerolog"
	"wellquite.org/golmdb"
)

// DataLoader handles loading data from MySQL to LMDB
type DataLoader struct {
	mysqlConn    *sql.DB
	lmdbClient   *golmdb.LMDBClient
	logger       zerolog.Logger
	mu           sync.RWMutex
	cachedTables map[string]bool // Track which tables are cached
}

// NewDataLoader creates a new data loader instance
func NewDataLoader(mysqlDSN string, lmdbPath string, logger zerolog.Logger) (*DataLoader, error) {
	// Connect to MySQL
	mysqlConn, err := sql.Open("mysql", mysqlDSN)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to MySQL: %w", err)
	}

	// Test MySQL connection
	if err := mysqlConn.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping MySQL: %w", err)
	}

	// Create LMDB client
	lmdbClient, err := golmdb.NewLMDB(
		logger,
		lmdbPath,
		0644,
		126,  // max readers
		50,   // max DBs
		0,    // environment flags
		1000, // batch size
	)
	if err != nil {
		mysqlConn.Close()
		return nil, fmt.Errorf("failed to create LMDB client: %w", err)
	}

	return &DataLoader{
		mysqlConn:    mysqlConn,
		lmdbClient:   lmdbClient,
		logger:       logger,
		cachedTables: make(map[string]bool),
	}, nil
}

// Close closes all connections
func (dl *DataLoader) Close() error {
	dl.lmdbClient.TerminateSync()
	return dl.mysqlConn.Close()
}

// LoadTableToLMDB loads a specific table from MySQL to LMDB
func (dl *DataLoader) LoadTableToLMDB(database, tableName string) error {
	dl.mu.Lock()
	defer dl.mu.Unlock()

	cacheKey := fmt.Sprintf("%s.%s", database, tableName)
	if dl.cachedTables[cacheKey] {
		log.Printf("Table %s already cached in LMDB", cacheKey)
		return nil
	}

	log.Printf("Loading table %s.%s from MySQL to LMDB", database, tableName)

	// Get table schema from MySQL
	schema, err := dl.getTableSchema(database, tableName)
	if err != nil {
		return fmt.Errorf("failed to get table schema: %w", err)
	}

	// Store schema in LMDB
	if err := dl.storeSchemaInLMDB(database, tableName, schema); err != nil {
		return fmt.Errorf("failed to store schema in LMDB: %w", err)
	}

	// Load data from MySQL
	query := fmt.Sprintf("SELECT * FROM `%s`.`%s`", database, tableName)
	rows, err := dl.mysqlConn.Query(query)
	if err != nil {
		return fmt.Errorf("failed to query MySQL table: %w", err)
	}
	defer rows.Close()

	// Store rows in LMDB
	count := 0
	return dl.lmdbClient.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("cached_tables", golmdb.Create)
		if err != nil {
			return err
		}

		for rows.Next() {
			// Scan row data
			scanDests := make([]interface{}, len(schema))
			for i := range scanDests {
				scanDests[i] = new(interface{})
			}

			if err := rows.Scan(scanDests...); err != nil {
				return fmt.Errorf("failed to scan row: %w", err)
			}

			// Convert to gmssql.Row
			row := make(gmssql.Row, len(schema))
			for i, dest := range scanDests {
				val := *(dest.(*interface{}))
				row[i] = val
			}

			// Store row in LMDB
			rowKey := fmt.Sprintf("%s:%s:row:%d", database, tableName, count)
			rowData, err := json.Marshal(row)
			if err != nil {
				return fmt.Errorf("failed to marshal row: %w", err)
			}

			if err := txn.Put(db, []byte(rowKey), rowData, 0); err != nil {
				return fmt.Errorf("failed to store row in LMDB: %w", err)
			}

			count++
			if count%1000 == 0 {
				log.Printf("Loaded %d rows from %s.%s", count, database, tableName)
			}
		}

		// Store row count
		countKey := fmt.Sprintf("%s:%s:count", database, tableName)
		if err := txn.Put(db, []byte(countKey), []byte(fmt.Sprintf("%d", count)), 0); err != nil {
			return fmt.Errorf("failed to store row count: %w", err)
		}

		dl.cachedTables[cacheKey] = true
		log.Printf("Successfully loaded %d rows from %s.%s to LMDB", count, database, tableName)
		return nil
	})
}

// getTableSchema retrieves the schema from MySQL
func (dl *DataLoader) getTableSchema(database, tableName string) (gmssql.Schema, error) {
	query := fmt.Sprintf(`
		SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA, 
		       CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE
		FROM INFORMATION_SCHEMA.COLUMNS 
		WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
		ORDER BY ORDINAL_POSITION
	`)

	rows, err := dl.mysqlConn.Query(query, database, tableName)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var schema gmssql.Schema
	for rows.Next() {
		var colName, dataType, isNullable, columnKey, extra string
		var charMaxLen, numPrecision, numScale sql.NullInt64

		err := rows.Scan(&colName, &dataType, &isNullable, &columnKey, &extra,
			&charMaxLen, &numPrecision, &numScale)
		if err != nil {
			return nil, err
		}

		// Convert MySQL data type to go-mysql-server type
		sqlType := dl.mysqlTypeToGMSType(dataType, charMaxLen.Int64, numPrecision.Int64, numScale.Int64)

		col := &gmssql.Column{
			Name:          colName,
			Type:          sqlType,
			Nullable:      isNullable == "YES",
			PrimaryKey:    columnKey == "PRI",
			AutoIncrement: extra == "auto_increment",
		}

		schema = append(schema, col)
	}

	return schema, nil
}

// mysqlTypeToGMSType converts MySQL data types to go-mysql-server types
func (dl *DataLoader) mysqlTypeToGMSType(mysqlType string, charMaxLen, numPrecision, numScale int64) gmssql.Type {
	switch mysqlType {
	case "int", "integer":
		return types.Int32
	case "bigint":
		return types.Int64
	case "smallint":
		return types.Int16
	case "tinyint":
		return types.Int8
	case "varchar":
		if charMaxLen > 0 {
			return types.MustCreateStringWithDefaults(sqltypes.VarChar, charMaxLen)
		}
		return types.MustCreateStringWithDefaults(sqltypes.VarChar, 255)
	case "char":
		if charMaxLen > 0 {
			return types.MustCreateStringWithDefaults(sqltypes.Char, charMaxLen)
		}
		return types.MustCreateStringWithDefaults(sqltypes.Char, 1)
	case "text", "mediumtext", "longtext":
		return types.Text
	case "decimal", "numeric":
		if numPrecision > 0 && numScale >= 0 {
			return types.MustCreateDecimalType(uint8(numPrecision), uint8(numScale))
		}
		return types.Float64
	case "float":
		return types.Float32
	case "double":
		return types.Float64
	case "date":
		return types.Date
	case "datetime":
		return types.Datetime
	case "timestamp":
		return types.Timestamp
	case "time":
		return types.Time
	case "blob", "mediumblob", "longblob":
		return types.Blob
	default:
		// Default to text for unknown types
		return types.Text
	}
}

// storeSchemaInLMDB stores the table schema in LMDB
func (dl *DataLoader) storeSchemaInLMDB(database, tableName string, schema gmssql.Schema) error {
	return dl.lmdbClient.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("cached_tables", golmdb.Create)
		if err != nil {
			return err
		}

		// Convert schema to JSON
		type columnDef struct {
			Name          string `json:"name"`
			Type          string `json:"type"`
			Nullable      bool   `json:"nullable"`
			PrimaryKey    bool   `json:"primary_key"`
			AutoIncrement bool   `json:"auto_increment"`
		}

		columns := make([]columnDef, len(schema))
		for i, col := range schema {
			columns[i] = columnDef{
				Name:          col.Name,
				Type:          col.Type.String(),
				Nullable:      col.Nullable,
				PrimaryKey:    col.PrimaryKey,
				AutoIncrement: col.AutoIncrement,
			}
		}

		schemaData, err := json.Marshal(columns)
		if err != nil {
			return err
		}

		schemaKey := fmt.Sprintf("%s:%s:schema", database, tableName)
		return txn.Put(db, []byte(schemaKey), schemaData, 0)
	})
}

// IsTableCached checks if a table is already cached in LMDB
func (dl *DataLoader) IsTableCached(database, tableName string) bool {
	dl.mu.RLock()
	defer dl.mu.RUnlock()

	cacheKey := fmt.Sprintf("%s.%s", database, tableName)
	return dl.cachedTables[cacheKey]
}

// RefreshTable forces a reload of a table from MySQL to LMDB
func (dl *DataLoader) RefreshTable(database, tableName string) error {
	dl.mu.Lock()
	cacheKey := fmt.Sprintf("%s.%s", database, tableName)
	delete(dl.cachedTables, cacheKey)
	dl.mu.Unlock()

	// Clear existing data in LMDB
	if err := dl.clearTableFromLMDB(database, tableName); err != nil {
		return fmt.Errorf("failed to clear table from LMDB: %w", err)
	}

	// Reload the table
	return dl.LoadTableToLMDB(database, tableName)
}

// clearTableFromLMDB removes all data for a table from LMDB
func (dl *DataLoader) clearTableFromLMDB(database, tableName string) error {
	return dl.lmdbClient.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef("cached_tables", golmdb.Create)
		if err != nil {
			return err
		}

		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		prefix := fmt.Sprintf("%s:%s:", database, tableName)
		keysToDelete := []string{}

		// Collect keys to delete
		key, _, err := cursor.First()
		for err == nil {
			keyStr := string(key)
			if len(keyStr) >= len(prefix) && keyStr[:len(prefix)] == prefix {
				keysToDelete = append(keysToDelete, keyStr)
			}
			key, _, err = cursor.Next()
		}

		// Delete collected keys
		for _, key := range keysToDelete {
			if err := txn.Delete(db, []byte(key), nil); err != nil {
				return err
			}
		}

		return nil
	})
}