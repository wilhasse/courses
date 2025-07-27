# Complete SQL Execution Example

This document traces through a complete SQL query execution to show how parsing, optimization, and storage interaction work together.

## Example Query

Let's trace this complex query:

```sql
SELECT u.name, p.name as product_name, p.price 
FROM users u 
JOIN products p ON u.id = p.user_id 
WHERE p.price > 50 AND u.name LIKE 'A%'
GROUP BY u.id, u.name
HAVING COUNT(*) >= 1
ORDER BY u.name
LIMIT 10;
```

## Phase 1: SQL Parsing

### Input: Raw SQL String
The SQL text is sent from MySQL client to our server.

### Vitess Parser Output
```go
// Simplified AST structure created by vitess parser
&sqlparser.Select{
    SelectExprs: []sqlparser.SelectExpr{
        &sqlparser.AliasedExpr{Expr: &sqlparser.ColName{Name: "name", Qualifier: "u"}},
        &sqlparser.AliasedExpr{
            Expr: &sqlparser.ColName{Name: "name", Qualifier: "p"},
            As:   "product_name",
        },
        &sqlparser.AliasedExpr{Expr: &sqlparser.ColName{Name: "price", Qualifier: "p"}},
    },
    From: []sqlparser.TableExpr{
        &sqlparser.JoinTableExpr{
            LeftExpr:  &sqlparser.AliasedTableExpr{Expr: "users", As: "u"},
            Join:      "join",
            RightExpr: &sqlparser.AliasedTableExpr{Expr: "products", As: "p"},
            Condition: &sqlparser.OnExpr{
                Expr: &sqlparser.ComparisonExpr{
                    Left:     &sqlparser.ColName{Name: "id", Qualifier: "u"},
                    Operator: "=",
                    Right:    &sqlparser.ColName{Name: "user_id", Qualifier: "p"},
                },
            },
        },
    },
    Where: &sqlparser.Where{
        Type: "where",
        Expr: &sqlparser.AndExpr{
            Left: &sqlparser.ComparisonExpr{
                Left:     &sqlparser.ColName{Name: "price", Qualifier: "p"},
                Operator: ">",
                Right:    &sqlparser.SQLVal{Type: sqlparser.IntVal, Val: "50"},
            },
            Right: &sqlparser.ComparisonExpr{
                Left:     &sqlparser.ColName{Name: "name", Qualifier: "u"},
                Operator: "like",
                Right:    &sqlparser.SQLVal{Type: sqlparser.StrVal, Val: "'A%'"},
            },
        },
    },
    GroupBy: []sqlparser.Expr{
        &sqlparser.ColName{Name: "id", Qualifier: "u"},
        &sqlparser.ColName{Name: "name", Qualifier: "u"},
    },
    Having: &sqlparser.Where{
        Type: "having",
        Expr: &sqlparser.ComparisonExpr{
            Left:     &sqlparser.FuncExpr{Name: "count", Exprs: []sqlparser.SelectExpr{...}},
            Operator: ">=",
            Right:    &sqlparser.SQLVal{Type: sqlparser.IntVal, Val: "1"},
        },
    },
    OrderBy: []sqlparser.Order{
        {Expr: &sqlparser.ColName{Name: "name", Qualifier: "u"}, Direction: "asc"},
    },
    Limit: &sqlparser.Limit{Rowcount: &sqlparser.SQLVal{Type: sqlparser.IntVal, Val: "10"}},
}
```

## Phase 2: Analysis and Plan Building

### Step 1: Resolve Table References
```go
// Analyzer calls our DatabaseProvider
db, err := provider.Database(ctx, "testdb")
usersTable, found, err := db.GetTableInsensitive(ctx, "users")
productsTable, found, err := db.GetTableInsensitive(ctx, "products")
```

### Step 2: Resolve Column References
```go
// Verify columns exist in their respective tables
// u.id, u.name -> users table schema
// p.user_id, p.name, p.price -> products table schema
```

### Step 3: Initial Plan Tree (Before Optimization)
```
Limit[10]
└── Sort[u.name ASC]
    └── Project[u.name, p.name as product_name, p.price]
        └── Having[COUNT(*) >= 1]
            └── GroupBy[u.id, u.name]
                └── Filter[p.price > 50 AND u.name LIKE 'A%']
                    └── InnerJoin[u.id = p.user_id]
                        ├── TableAlias[u] → ResolvedTable[users]
                        └── TableAlias[p] → ResolvedTable[products]
```

## Phase 3: Optimization

### Step 1: Predicate Pushdown
Move WHERE conditions closer to tables:

```
// Before: Filter after Join
InnerJoin[u.id = p.user_id]
├── ResolvedTable[users]
└── ResolvedTable[products]
↓ Filter[p.price > 50 AND u.name LIKE 'A%']

// After: Filters pushed down
InnerJoin[u.id = p.user_id]
├── Filter[u.name LIKE 'A%']
│   └── ResolvedTable[users]
└── Filter[p.price > 50]
    └── ResolvedTable[products]
```

### Step 2: Join Reordering (if needed)
Based on table statistics, optimizer might reorder joins.

### Step 3: Final Optimized Plan
```
Limit[10]
└── Sort[u.name ASC]
    └── Project[u.name, p.name as product_name, p.price]
        └── Having[COUNT(*) >= 1]
            └── GroupBy[u.id, u.name]
                └── InnerJoin[u.id = p.user_id]
                    ├── Filter[u.name LIKE 'A%']
                    │   └── ResolvedTable[users]
                    └── Filter[p.price > 50]
                        └── ResolvedTable[products]
```

## Phase 4: Execution

### Step 1: Table Scans with Filters

#### Users Table Scan:
```go
// Call to our storage layer
func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    // If we implemented WithFilters, the filter would be pushed here
    rows, err := t.storage.GetRows(t.database, t.name)
    
    // Returns iterator that yields:
    // Row 1: [1, "Alice", "alice@example.com", "2023-01-01"]
    // Row 2: [2, "Bob", "bob@example.com", "2023-01-02"]
    // (but Bob would be filtered out by u.name LIKE 'A%')
}
```

**Filter Applied:** `u.name LIKE 'A%'`
- Row 1: Alice ✓ (matches)
- Row 2: Bob ✗ (filtered out)

**Result:** `[(1, "Alice", "alice@example.com", "2023-01-01")]`

#### Products Table Scan:
```go
// Call to our storage layer
func (t *Table) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
    rows, err := t.storage.GetRows(t.database, t.name)
    
    // Returns iterator that yields:
    // Row 1: [1, "Laptop", 999.99, "Electronics"]
    // Row 2: [2, "Book", 19.99, "Education"]  
    // Row 3: [3, "Coffee Mug", 12.50, "Kitchen"]
}
```

**Filter Applied:** `p.price > 50`
- Row 1: Laptop (999.99) ✓ (matches)
- Row 2: Book (19.99) ✗ (filtered out)
- Row 3: Coffee Mug (12.50) ✗ (filtered out)

**Result:** `[(1, "Laptop", 999.99, "Electronics")]`

### Step 2: Join Execution

#### Nested Loop Join Process:
```go
// For each row from users (left side):
leftRow := [1, "Alice", "alice@example.com", "2023-01-01"]

// Scan products (right side) looking for matches:
for rightRow := range productsRows {
    // rightRow: [1, "Laptop", 999.99, "Electronics"]
    
    // Test join condition: u.id = p.user_id
    // leftRow[0] (u.id=1) == rightRow[0] (p.user_id=1) ✓
    
    combinedRow := [1, "Alice", "alice@example.com", "2023-01-01", 1, "Laptop", 999.99, "Electronics"]
    // Emit this joined row
}
```

**Join Result:** 
```
[(1, "Alice", "alice@example.com", "2023-01-01", 1, "Laptop", 999.99, "Electronics")]
```

### Step 3: Grouping

#### Hash-based Grouping:
```go
// Group by: u.id, u.name
groupKey := "1|Alice"  // Combine u.id=1 and u.name="Alice"

groups := map[string]*AggregateState{
    "1|Alice": {
        groupValues: [1, "Alice"],
        aggregators: [CountAggregator{count: 1}],
        firstRow: [1, "Alice", "alice@example.com", "2023-01-01", 1, "Laptop", 999.99, "Electronics"]
    }
}
```

### Step 4: Having Filter

#### Apply HAVING condition:
```go
// Test: COUNT(*) >= 1
// For group "1|Alice": count = 1
// 1 >= 1 ✓ (group passes)
```

**Result after HAVING:** 
```
[(1, "Alice", 1)]  // u.id, u.name, COUNT(*)
```

### Step 5: Projection

#### Select specific columns:
```sql
-- SELECT u.name, p.name as product_name, p.price
```

```go
// From grouped result, need to access original joined row
// This requires storing additional data in grouping phase
projectedRow := ["Alice", "Laptop", 999.99]
```

### Step 6: Sorting

#### ORDER BY u.name:
```go
// Single row, so no sorting needed
// With multiple rows, would use sort.Slice()
```

### Step 7: Limit

#### LIMIT 10:
```go
// Take first 10 rows (we only have 1)
finalResult := [["Alice", "Laptop", 999.99]]
```

## Storage Interface Calls Summary

During this query execution, our storage layer was called:

1. **Table Resolution:**
   - `DatabaseProvider.Database("testdb")`
   - `Database.GetTableInsensitive("users")`
   - `Database.GetTableInsensitive("products")`

2. **Schema Information:**
   - `Table.Schema()` for both tables

3. **Data Scanning:**
   - `Table.Partitions()` for both tables
   - `Table.PartitionRows()` for both tables
   - `Storage.GetRows("testdb", "users")`
   - `Storage.GetRows("testdb", "products")`

4. **Row Iteration:**
   - Multiple calls to `RowIter.Next()` for both tables

## Performance Optimizations Applied

1. **Predicate Pushdown:** Filters moved to table scan level
2. **Early Filtering:** Rows eliminated before expensive join
3. **Hash Grouping:** Efficient grouping for aggregation
4. **Streaming:** Results processed row-by-row, not materialized in memory

## What Your Storage Layer Needs to Implement

**Minimum Required:**
- `Storage.GetRows()` - Basic row scanning
- `Table.Schema()` - Column definitions
- `Table.Partitions()` / `Table.PartitionRows()` - Data access

**For Better Performance:**
- `sql.FilteredTable` - Push filters to storage
- `sql.ProjectedTable` - Push column selection to storage  
- `sql.IndexedTable` - Use indexes for faster access
- `sql.StatisticsTable` - Provide stats for optimization

This shows how go-mysql-server handles all the complex SQL processing while your storage layer only needs to provide simple row access and schema information!