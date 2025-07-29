# Hybrid Query System - Implementation Details

This document explains the key parts of the hybrid query system implementation, showing how it loads data from MySQL to LMDB and performs cross-source joins.

## 1. Data Loading Process

### Loading a Table from MySQL to LMDB

The `LoadTableToLMDB` function in `data_loader.go` handles copying data from MySQL to LMDB:

```go
func (dl *DataLoader) LoadTableToLMDB(database, tableName string) error {
    // Step 1: Get table schema from MySQL
    schema, err := dl.getTableSchema(database, tableName)
    
    // Step 2: Store schema in LMDB
    if err := dl.storeSchemaInLMDB(database, tableName, schema); err != nil {
        return err
    }
    
    // Step 3: Query all data from MySQL table
    query := fmt.Sprintf("SELECT * FROM `%s`.`%s`", database, tableName)
    rows, err := dl.mysqlConn.Query(query)
    
    // Step 4: Store each row in LMDB
    return dl.lmdbClient.Update(func(txn *golmdb.ReadWriteTxn) error {
        for rows.Next() {
            // Scan row data
            scanDests := make([]interface{}, len(schema))
            rows.Scan(scanDests...)
            
            // Store in LMDB with key format: database:table:row:id
            rowKey := fmt.Sprintf("%s:%s:row:%d", database, tableName, count)
            rowData, _ := json.Marshal(row)
            txn.Put(db, []byte(rowKey), rowData, 0)
        }
    })
}
```

### Schema Extraction from MySQL

The schema is extracted using MySQL's INFORMATION_SCHEMA:

```go
func (dl *DataLoader) getTableSchema(database, tableName string) (gmssql.Schema, error) {
    query := `
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, EXTRA
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION
    `
    
    rows, err := dl.mysqlConn.Query(query, database, tableName)
    
    // Convert MySQL types to go-mysql-server types
    for rows.Next() {
        var colName, dataType string
        rows.Scan(&colName, &dataType, ...)
        
        sqlType := dl.mysqlTypeToGMSType(dataType, ...)
        col := &gmssql.Column{
            Name:       colName,
            Type:       sqlType,
            PrimaryKey: columnKey == "PRI",
        }
        schema = append(schema, col)
    }
}
```

## 2. Query Analysis and Detection

### Parsing SQL Queries

The `AnalyzeQuery` function in `sql_parser.go` uses the Vitess SQL parser to analyze queries:

```go
func (p *SQLParser) AnalyzeQuery(query string, currentDatabase string) (*QueryAnalysis, error) {
    // Parse SQL using Vitess parser
    stmt, err := sqlparser.Parse(query)
    
    // Extract tables from FROM clause
    tables := p.extractTablesFromTableExprs(stmt.From, currentDatabase)
    
    // Classify each table as cached or remote
    for _, table := range tables {
        if p.isTableCached(table.Database, table.Table) {
            analysis.HasCachedTable = true
            analysis.CachedTables = append(analysis.CachedTables, table)
        } else {
            analysis.RemoteTables = append(analysis.RemoteTables, table)
        }
    }
    
    // Check if query needs rewriting (has both cached and remote tables)
    analysis.RequiresRewrite = analysis.HasCachedTable && len(analysis.RemoteTables) > 0
}
```

### Detecting ACORDO_GM References

The system checks if a table is cached (like ACORDO_GM):

```go
func (p *SQLParser) isTableCached(database, table string) bool {
    // Special case: always consider ACORDO_GM as cached
    if strings.ToLower(table) == "acordo_gm" {
        return true
    }
    
    // Check registered cached tables
    key := fmt.Sprintf("%s.%s", strings.ToLower(database), strings.ToLower(table))
    return p.cachedTables[key]
}
```

### Extracting Join Conditions

Join conditions are extracted for later use in cross-source joins:

```go
func (p *SQLParser) extractJoinConditions(tableExprs sqlparser.TableExprs, where *sqlparser.Where) []JoinCondition {
    // Extract from explicit JOIN ON clauses
    for _, tableExpr := range tableExprs {
        if join, ok := tableExpr.(*sqlparser.JoinTableExpr); ok {
            if join.Condition.On != nil {
                // Extract conditions like: a.id = b.acordo_id
                conditions = append(conditions, p.extractConditionsFromExpr(join.Condition.On)...)
            }
        }
    }
    
    // Also extract from WHERE clause (implicit joins)
    if where != nil {
        conditions = append(conditions, p.extractConditionsFromExpr(where.Expr)...)
    }
}
```

## 3. Query Rewriting

### Removing Cached Tables from Queries

The `RewriteQuery` function in `query_rewriter.go` removes cached tables:

```go
func (r *QueryRewriter) RewriteQuery(query string, currentDatabase string) (*RewriteResult, error) {
    // Original query:
    // SELECT a.*, b.* FROM ACORDO_GM a JOIN transactions b ON a.id = b.acordo_id
    
    // Rewrite to remove ACORDO_GM:
    rewrittenStmt := r.rewriteSelectStatement(selectStmt, analysis, currentDatabase)
    
    // Result: SELECT b.* FROM transactions b
}
```

### Rewriting Table Expressions

The FROM clause is rewritten to remove cached tables:

```go
func (r *QueryRewriter) rewriteTableExprs(exprs sqlparser.TableExprs, analysis *QueryAnalysis) sqlparser.TableExprs {
    var rewritten sqlparser.TableExprs
    
    for _, expr := range exprs {
        // Check if table is cached
        if tableName, ok := expr.(*sqlparser.AliasedTableExpr); ok {
            table := tableName.Expr.(sqlparser.TableName).Name.String()
            
            // Skip cached tables like ACORDO_GM
            for _, cached := range analysis.CachedTables {
                if cached.Table == table {
                    continue // Remove from query
                }
            }
        }
        rewritten = append(rewritten, expr)
    }
}
```

## 4. Cross-Source Join Execution

### Executing Hybrid Queries

The `ExecuteHybridQuery` function in `join_executor.go` coordinates execution:

```go
func (je *JoinExecutor) ExecuteHybridQuery(rewriteResult *RewriteResult, currentDatabase string) (*QueryResult, error) {
    // Step 1: Execute rewritten query on MySQL (without ACORDO_GM)
    remoteResult, err := je.executeRemoteQuery(rewriteResult.RemoteQuery)
    
    // Step 2: Get ACORDO_GM data from LMDB cache
    cachedResults := make(map[string]*QueryResult)
    for _, tableName := range rewriteResult.CachedTableNames {
        cachedResult, err := je.getCachedTableData(database, table)
        cachedResults[tableName] = cachedResult
    }
    
    // Step 3: Join results from both sources
    return je.performJoin(remoteResult, cachedResults, rewriteResult.JoinStrategy)
}
```

### Retrieving Cached Data from LMDB

```go
func (je *JoinExecutor) getCachedTableData(database, table string) (*QueryResult, error) {
    err := je.lmdbClient.View(func(txn *golmdb.ReadOnlyTxn) error {
        // Get schema first
        schemaKey := fmt.Sprintf("%s:%s:schema", database, table)
        schemaData, _ := txn.Get(db, []byte(schemaKey))
        
        // Get all rows with prefix matching
        prefix := fmt.Sprintf("%s:%s:row:", database, table)
        cursor, _ := txn.NewCursor(db)
        
        key, data, _ := cursor.First()
        for err == nil {
            if strings.HasPrefix(string(key), prefix) {
                var row gmssql.Row
                json.Unmarshal(data, &row)
                result.Rows = append(result.Rows, row)
            }
            key, data, err = cursor.Next()
        }
    })
}
```

### Performing In-Memory Joins

The join logic handles combining results from different sources:

```go
func (je *JoinExecutor) performConditionalJoin(left, right *QueryResult, conditions []JoinCondition) (*QueryResult, error) {
    // Create column index maps for fast lookup
    leftColIndex := make(map[string]int)
    for i, col := range left.Columns {
        leftColIndex[col] = i
    }
    
    // Nested loop join implementation
    for _, leftRow := range left.Rows {
        for _, rightRow := range right.Rows {
            // Check join conditions
            match := true
            for _, condition := range conditions {
                // Get column values
                leftVal := leftRow[leftColIndex[condition.LeftColumn]]
                rightVal := rightRow[rightColIndex[condition.RightColumn]]
                
                // Compare based on operator (=, <, >, etc.)
                if !je.compareValues(leftVal, rightVal, condition.Operator) {
                    match = false
                    break
                }
            }
            
            if match {
                // Combine rows from both sources
                combinedRow := append(leftRow, rightRow...)
                rows = append(rows, combinedRow)
            }
        }
    }
}
```

## 5. Integration with go-mysql-server

### Wrapping Tables for Transparent Access

The `HybridTable` in `example_integration.go` shows how to integrate:

```go
type HybridTable struct {
    name          string
    schema        gmssql.Schema
    handler       *HybridHandler
    fallbackTable gmssql.Table
}

func (t *HybridTable) PartitionRows(ctx *gmssql.Context, partition gmssql.Partition) (gmssql.RowIter, error) {
    // Check if this table is cached
    if t.handler.IsTableCached(t.database, t.name) {
        // Return data from LMDB cache
        rows, _, _ := t.handler.GetCachedTableData(t.database, t.name)
        return &hybridRowIter{rows: rows}, nil
    }
    
    // Fall back to original table
    return t.fallbackTable.PartitionRows(ctx, partition)
}
```

## Key Design Decisions

1. **LMDB Key Structure**: Uses hierarchical keys like `database:table:row:id` for efficient prefix scanning
2. **JSON Serialization**: Rows and schemas are stored as JSON for flexibility
3. **Lazy Loading**: Tables are loaded on-demand rather than at startup
4. **Fallback Strategy**: If cache access fails, queries fall back to MySQL
5. **In-Memory Joins**: Cross-source joins are performed in memory using nested loops

## Performance Optimizations

1. **Batch Loading**: Data is loaded in transactions to minimize LMDB overhead
2. **Cursor-Based Iteration**: Uses LMDB cursors for efficient data retrieval
3. **Schema Caching**: Table schemas are cached to avoid repeated lookups
4. **Selective Column Loading**: Future optimization could load only needed columns

## Example Query Flow

For the query: `SELECT * FROM ACORDO_GM a JOIN transactions b ON a.id = b.acordo_id`

1. **Parse**: Identify ACORDO_GM as cached, transactions as remote
2. **Rewrite**: Create query `SELECT * FROM transactions b` for MySQL
3. **Execute**: 
   - Run rewritten query on MySQL
   - Fetch ACORDO_GM data from LMDB
4. **Join**: Perform in-memory join on `a.id = b.acordo_id`
5. **Return**: Combined result set to the client

This architecture allows transparent caching of frequently accessed tables while maintaining query compatibility.