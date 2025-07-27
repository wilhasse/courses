# Storage Interface and Row Scanning

## How go-mysql-server Calls Your Storage

### 1. Table Resolution
When the analyzer encounters a table reference, it calls your DatabaseProvider:

```go
// In our example: pkg/provider/database_provider.go
func (p *DatabaseProvider) Database(ctx *sql.Context, name string) (sql.Database, error) {
    // Returns our custom Database implementation
}

// Then calls our Database
func (db *Database) GetTableInsensitive(ctx *sql.Context, tblName string) (sql.Table, bool, error) {
    // Returns our custom Table implementation
}
```

### 2. Row Scanning Process

go-mysql-server uses a **partition-based scanning model**:

```go
// Step 1: Get partitions
partitions, err := table.Partitions(ctx)

// Step 2: For each partition, get rows
for partition := range partitions {
    rowIter, err := table.PartitionRows(ctx, partition)
    
    // Step 3: Iterate through rows
    for {
        row, err := rowIter.Next(ctx)
        if err == io.EOF {
            break
        }
        // Process row...
    }
}
```

### 3. Our Storage Implementation

In our example (`pkg/provider/table.go`):

```go
// Partitions - We use a single partition for simplicity
func (t *Table) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
    return &singlePartitionIter{}, nil
}

// PartitionRows - This is where we call our storage backend
func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    // Call our custom storage
    rows, err := t.storage.GetRows(t.database, t.name)
    if err != nil {
        return nil, err
    }
    // Return iterator over our rows
    return &tableRowIter{rows: rows, index: 0}, nil
}
```

### 4. Row Iterator Implementation

```go
type tableRowIter struct {
    rows  []sql.Row  // Data from our storage
    index int        // Current position
}

func (t *tableRowIter) Next(ctx *sql.Context) (sql.Row, error) {
    if t.index >= len(t.rows) {
        return nil, io.EOF  // Signal end of data
    }
    row := t.rows[t.index]
    t.index++
    return row, nil
}
```

## Partition Strategy for Large Datasets

For production systems, you'd implement intelligent partitioning:

```go
// Example: Date-based partitioning
type DatePartition struct {
    startDate time.Time
    endDate   time.Time
}

func (t *Table) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
    // Return partitions based on date ranges
    partitions := []sql.Partition{
        &DatePartition{startDate: jan1, endDate: jan31},
        &DatePartition{startDate: feb1, endDate: feb28},
        // ...
    }
    return &partitionIter{partitions: partitions}, nil
}

func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    datePartition := partition.(*DatePartition)
    // Only scan rows within this date range
    rows := t.storage.GetRowsInDateRange(datePartition.startDate, datePartition.endDate)
    return &tableRowIter{rows: rows}, nil
}
```

## Push-Down Optimizations

go-mysql-server can push filters down to your storage layer:

```go
// Implement sql.FilteredTable interface
func (t *Table) WithFilters(ctx *sql.Context, filters []sql.Expression) sql.Table {
    // Create a new table instance with filters
    return &FilteredTable{
        Table:   t,
        filters: filters,
    }
}

func (ft *FilteredTable) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    // Apply filters at storage level instead of in memory
    // Example: Convert "price > 50" to storage-specific filter
    storageFilters := convertFiltersToStorage(ft.filters)
    rows := ft.storage.GetRowsWithFilters(ft.database, ft.name, storageFilters)
    return &tableRowIter{rows: rows}, nil
}
```