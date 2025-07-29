package hybrid

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"strings"

	gmssql "github.com/dolthub/go-mysql-server/sql"
	"wellquite.org/golmdb"
)

// JoinExecutor handles joining results from MySQL and LMDB
type JoinExecutor struct {
	mysqlConn  *sql.DB
	lmdbClient *golmdb.LMDBClient
}

// NewJoinExecutor creates a new join executor
func NewJoinExecutor(mysqlConn *sql.DB, lmdbClient *golmdb.LMDBClient) *JoinExecutor {
	return &JoinExecutor{
		mysqlConn:  mysqlConn,
		lmdbClient: lmdbClient,
	}
}

// QueryResult represents the result of a query from either source
type QueryResult struct {
	Columns []string
	Rows    [][]interface{}
}

// ExecuteHybridQuery executes a query across MySQL and LMDB sources and joins the results
func (je *JoinExecutor) ExecuteHybridQuery(rewriteResult *RewriteResult, currentDatabase string) (*QueryResult, error) {
	// If no cached tables, just execute on MySQL
	if len(rewriteResult.CachedTableNames) == 0 {
		return je.executeRemoteQuery(rewriteResult.RemoteQuery)
	}

	// Execute remote query on MySQL (if there are remote tables)
	var remoteResult *QueryResult
	var err error
	if rewriteResult.RemoteQuery != "" && !strings.Contains(strings.ToLower(rewriteResult.RemoteQuery), "select 1 as dummy") {
		remoteResult, err = je.executeRemoteQuery(rewriteResult.RemoteQuery)
		if err != nil {
			return nil, fmt.Errorf("failed to execute remote query: %w", err)
		}
	}

	// Get data from cached tables in LMDB
	cachedResults := make(map[string]*QueryResult)
	for _, tableName := range rewriteResult.CachedTableNames {
		// Parse database and table name
		parts := strings.Split(tableName, ".")
		database := currentDatabase
		table := tableName
		if len(parts) == 2 {
			database = parts[0]
			table = parts[1]
		}

		cachedResult, err := je.getCachedTableData(database, table)
		if err != nil {
			return nil, fmt.Errorf("failed to get cached data for %s: %w", tableName, err)
		}
		cachedResults[tableName] = cachedResult
	}

	// If only cached tables, return the first one (for simple queries)
	if remoteResult == nil && len(cachedResults) == 1 {
		for _, result := range cachedResults {
			return result, nil
		}
	}

	// Perform join between remote and cached results
	if remoteResult != nil && len(cachedResults) > 0 {
		return je.performJoin(remoteResult, cachedResults, rewriteResult.JoinStrategy)
	}

	// Handle multiple cached tables join (not implemented yet)
	return nil, fmt.Errorf("joining multiple cached tables is not yet implemented")
}

// executeRemoteQuery executes a query on the remote MySQL server
func (je *JoinExecutor) executeRemoteQuery(query string) (*QueryResult, error) {
	log.Printf("Executing remote query: %s", query)

	rows, err := je.mysqlConn.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Get column information
	columns, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	result := &QueryResult{
		Columns: columns,
		Rows:    make([][]interface{}, 0),
	}

	// Scan all rows
	for rows.Next() {
		scanDests := make([]interface{}, len(columns))
		for i := range scanDests {
			scanDests[i] = new(interface{})
		}

		if err := rows.Scan(scanDests...); err != nil {
			return nil, err
		}

		row := make([]interface{}, len(columns))
		for i, dest := range scanDests {
			row[i] = *(dest.(*interface{}))
		}
		result.Rows = append(result.Rows, row)
	}

	log.Printf("Remote query returned %d rows", len(result.Rows))
	return result, nil
}

// getCachedTableData retrieves data for a cached table from LMDB
func (je *JoinExecutor) getCachedTableData(database, table string) (*QueryResult, error) {
	log.Printf("Getting cached data for %s.%s", database, table)

	var result *QueryResult

	err := je.lmdbClient.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("cached_tables", 0)
		if err != nil {
			return err
		}

		// Get schema
		schemaKey := fmt.Sprintf("%s:%s:schema", database, table)
		schemaData, err := txn.Get(db, []byte(schemaKey))
		if err != nil {
			return fmt.Errorf("schema not found for cached table %s.%s", database, table)
		}

		// Parse schema
		type columnDef struct {
			Name          string `json:"name"`
			Type          string `json:"type"`
			Nullable      bool   `json:"nullable"`
			PrimaryKey    bool   `json:"primary_key"`
			AutoIncrement bool   `json:"auto_increment"`
		}

		var columns []columnDef
		if err := json.Unmarshal(schemaData, &columns); err != nil {
			return err
		}

		// Build column names
		columnNames := make([]string, len(columns))
		for i, col := range columns {
			columnNames[i] = col.Name
		}

		result = &QueryResult{
			Columns: columnNames,
			Rows:    make([][]interface{}, 0),
		}

		// Get all rows
		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		prefix := fmt.Sprintf("%s:%s:row:", database, table)
		key, data, err := cursor.First()
		for err == nil {
			keyStr := string(key)
			if strings.HasPrefix(keyStr, prefix) {
				var row gmssql.Row
				if err := json.Unmarshal(data, &row); err == nil {
					// Convert gmssql.Row to []interface{}
					rowData := make([]interface{}, len(row))
					for i, val := range row {
						rowData[i] = val
					}
					result.Rows = append(result.Rows, rowData)
				}
			}
			key, data, err = cursor.Next()
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	log.Printf("Cached table %s.%s has %d rows", database, table, len(result.Rows))
	return result, nil
}

// performJoin performs a join between remote and cached results
func (je *JoinExecutor) performJoin(remoteResult *QueryResult, cachedResults map[string]*QueryResult, strategy JoinStrategy) (*QueryResult, error) {
	// For simplicity, we'll implement a nested loop join
	// In production, you'd want to use more efficient join algorithms based on data size

	// For now, assume we're joining with one cached table
	if len(cachedResults) != 1 {
		return nil, fmt.Errorf("currently only support joining with one cached table")
	}

	var cachedTableName string
	var cachedResult *QueryResult
	for name, result := range cachedResults {
		cachedTableName = name
		cachedResult = result
		break
	}

	log.Printf("Performing join between remote result (%d rows) and cached table %s (%d rows)",
		len(remoteResult.Rows), cachedTableName, len(cachedResult.Rows))
	log.Printf("Join conditions: %+v", strategy.Conditions)

	// Find join columns
	if len(strategy.Conditions) == 0 {
		log.Printf("WARNING: No join conditions found, would produce cartesian product")
		// For now, return an error instead of cartesian product
		return nil, fmt.Errorf("no join conditions found - this would produce a cartesian product")
	}

	// Log column information for debugging
	log.Printf("Remote result columns: %v", remoteResult.Columns)
	log.Printf("Cached result columns: %v", cachedResult.Columns)
	
	// Perform join based on conditions
	return je.performConditionalJoin(remoteResult, cachedResult, strategy.Conditions)
}

// cartesianProduct performs a cartesian product between two results
func (je *JoinExecutor) cartesianProduct(left, right *QueryResult) (*QueryResult, error) {
	// Combine columns
	columns := append(left.Columns, right.Columns...)

	// Generate all combinations
	rows := make([][]interface{}, 0, len(left.Rows)*len(right.Rows))
	for _, leftRow := range left.Rows {
		for _, rightRow := range right.Rows {
			combinedRow := append(leftRow, rightRow...)
			rows = append(rows, combinedRow)
		}
	}

	return &QueryResult{
		Columns: columns,
		Rows:    rows,
	}, nil
}

// performConditionalJoin performs a join based on conditions
func (je *JoinExecutor) performConditionalJoin(left, right *QueryResult, conditions []JoinCondition) (*QueryResult, error) {
	// Create column index maps
	leftColIndex := make(map[string]int)
	for i, col := range left.Columns {
		leftColIndex[col] = i
	}

	rightColIndex := make(map[string]int)
	for i, col := range right.Columns {
		rightColIndex[col] = i
	}

	// Combine columns
	columns := append(left.Columns, right.Columns...)
	rows := make([][]interface{}, 0)

	// Nested loop join with condition checking
	for _, leftRow := range left.Rows {
		for _, rightRow := range right.Rows {
			// Check if all join conditions are satisfied
			match := true
			for _, condition := range conditions {
				// Find column indices
				leftIdx, leftOk := leftColIndex[condition.LeftColumn]
				rightIdx, rightOk := rightColIndex[condition.RightColumn]

				if !leftOk || !rightOk {
					// Try swapping if the condition was specified in reverse
					leftIdx, leftOk = leftColIndex[condition.RightColumn]
					rightIdx, rightOk = rightColIndex[condition.LeftColumn]
				}

				if !leftOk || !rightOk {
					// Skip this condition if columns not found
					continue
				}

				// Compare values
				if !je.compareValues(leftRow[leftIdx], rightRow[rightIdx], condition.Operator) {
					match = false
					break
				}
			}

			if match {
				combinedRow := append(leftRow, rightRow...)
				rows = append(rows, combinedRow)
			}
		}
	}

	log.Printf("Join produced %d rows", len(rows))

	return &QueryResult{
		Columns: columns,
		Rows:    rows,
	}, nil
}

// compareValues compares two values based on the operator
func (je *JoinExecutor) compareValues(left, right interface{}, operator string) bool {
	// Handle NULL values
	if left == nil || right == nil {
		return left == nil && right == nil && operator == "="
	}

	// Convert to comparable types
	leftStr := fmt.Sprintf("%v", left)
	rightStr := fmt.Sprintf("%v", right)

	switch operator {
	case "=":
		// Try numeric comparison first
		if leftNum, leftErr := toFloat64(left); leftErr == nil {
			if rightNum, rightErr := toFloat64(right); rightErr == nil {
				return leftNum == rightNum
			}
		}
		// Fall back to string comparison
		return leftStr == rightStr
	case "!=", "<>":
		return leftStr != rightStr
	case "<":
		if leftNum, leftErr := toFloat64(left); leftErr == nil {
			if rightNum, rightErr := toFloat64(right); rightErr == nil {
				return leftNum < rightNum
			}
		}
		return leftStr < rightStr
	case "<=":
		if leftNum, leftErr := toFloat64(left); leftErr == nil {
			if rightNum, rightErr := toFloat64(right); rightErr == nil {
				return leftNum <= rightNum
			}
		}
		return leftStr <= rightStr
	case ">":
		if leftNum, leftErr := toFloat64(left); leftErr == nil {
			if rightNum, rightErr := toFloat64(right); rightErr == nil {
				return leftNum > rightNum
			}
		}
		return leftStr > rightStr
	case ">=":
		if leftNum, leftErr := toFloat64(left); leftErr == nil {
			if rightNum, rightErr := toFloat64(right); rightErr == nil {
				return leftNum >= rightNum
			}
		}
		return leftStr >= rightStr
	default:
		return false
	}
}

// toFloat64 attempts to convert a value to float64
func toFloat64(val interface{}) (float64, error) {
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(v.Uint()), nil
	case reflect.Float32, reflect.Float64:
		return v.Float(), nil
	default:
		return 0, fmt.Errorf("cannot convert to float64")
	}
}