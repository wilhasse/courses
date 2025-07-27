# Joins and Aggregations in go-mysql-server

## How Join Operations Work

### 1. Join Planning

When the analyzer encounters a JOIN, it creates join plan nodes:

```sql
SELECT u.name, p.name 
FROM users u 
JOIN products p ON u.id = p.user_id
```

**Plan Tree:**
```
Project[u.name, p.name]
└── InnerJoin[u.id = p.user_id]
    ├── TableAlias[u] → ResolvedTable[users]
    └── TableAlias[p] → ResolvedTable[products]
```

### 2. Join Execution Strategies

go-mysql-server supports multiple join algorithms:

#### A. Nested Loop Join (Default for small tables)
```go
type NestedLoopJoin struct {
    left  sql.Node
    right sql.Node
    cond  sql.Expression
}

func (j *NestedLoopJoin) RowIter(ctx *sql.Context, row sql.Row) (sql.RowIter, error) {
    return &nestedLoopJoinIter{
        left:      j.left,
        right:     j.right,
        condition: j.cond,
    }, nil
}

// Execution: For each left row, scan all right rows
func (iter *nestedLoopJoinIter) Next(ctx *sql.Context) (sql.Row, error) {
    for {
        if iter.rightIter == nil {
            // Get next left row
            leftRow, err := iter.leftIter.Next(ctx)
            if err != nil {
                return nil, err
            }
            iter.currentLeft = leftRow
            // Start new right scan
            iter.rightIter, _ = iter.right.RowIter(ctx, nil)
        }
        
        // Get next right row
        rightRow, err := iter.rightIter.Next(ctx)
        if err == io.EOF {
            // Finished right scan, move to next left row
            iter.rightIter = nil
            continue
        }
        
        // Combine and test join condition
        combinedRow := append(iter.currentLeft, rightRow...)
        match, err := iter.condition.Eval(ctx, combinedRow)
        if err != nil {
            return nil, err
        }
        if match == true {
            return combinedRow, nil
        }
    }
}
```

#### B. Hash Join (For larger tables)
```go
type HashJoin struct {
    left     sql.Node
    right    sql.Node
    leftKey  sql.Expression  // u.id
    rightKey sql.Expression  // p.user_id
}

func (j *HashJoin) RowIter(ctx *sql.Context, row sql.Row) (sql.RowIter, error) {
    // Phase 1: Build hash table from smaller relation (left)
    hashTable := make(map[interface{}][]sql.Row)
    leftIter, _ := j.left.RowIter(ctx, nil)
    
    for {
        leftRow, err := leftIter.Next(ctx)
        if err == io.EOF {
            break
        }
        
        // Hash the join key
        key, _ := j.leftKey.Eval(ctx, leftRow)
        hashTable[key] = append(hashTable[key], leftRow)
    }
    
    // Phase 2: Probe with right relation
    return &hashJoinIter{
        hashTable: hashTable,
        rightIter: j.right.RowIter(ctx, nil),
        rightKey:  j.rightKey,
    }, nil
}

func (iter *hashJoinIter) Next(ctx *sql.Context) (sql.Row, error) {
    for {
        if iter.currentMatches == nil {
            // Get next right row
            rightRow, err := iter.rightIter.Next(ctx)
            if err != nil {
                return nil, err
            }
            
            // Look up in hash table
            key, _ := iter.rightKey.Eval(ctx, rightRow)
            iter.currentMatches = iter.hashTable[key]
            iter.currentRight = rightRow
            iter.matchIndex = 0
        }
        
        if iter.matchIndex >= len(iter.currentMatches) {
            // No more matches for this right row
            iter.currentMatches = nil
            continue
        }
        
        // Return matched pair
        leftRow := iter.currentMatches[iter.matchIndex]
        iter.matchIndex++
        return append(leftRow, iter.currentRight...), nil
    }
}
```

### 3. Join Types Implementation

#### Inner Join
```go
// Only returns rows where join condition is true
if joinConditionResult == true {
    return combinedRow, nil
}
```

#### Left Outer Join
```go
// Returns all left rows, null-padded right rows for non-matches
if joinConditionResult == true {
    return combinedRow, nil
} else if noMoreRightRows {
    nullPaddedRow := append(leftRow, makeNullRow(rightSchema)...)
    return nullPaddedRow, nil
}
```

#### Cross Join (Cartesian Product)
```go
// No join condition, returns all combinations
return append(leftRow, rightRow...), nil
```

## How Aggregation Operations Work

### 1. Aggregation Planning

```sql
SELECT u.id, COUNT(*), AVG(p.price)
FROM users u
JOIN products p ON u.id = p.user_id
GROUP BY u.id
HAVING COUNT(*) > 1
```

**Plan Tree:**
```
Project[u.id, COUNT(*), AVG(p.price)]
└── Having[COUNT(*) > 1]
    └── GroupBy[u.id, COUNT(*), AVG(p.price)]
        └── InnerJoin[u.id = p.user_id]
            ├── ResolvedTable[users]
            └── ResolvedTable[products]
```

### 2. Aggregation Execution

#### A. Hash-Based Grouping
```go
type GroupBy struct {
    child       sql.Node
    groupExprs  []sql.Expression  // GROUP BY columns
    aggExprs    []sql.Expression  // Aggregate functions
}

func (g *GroupBy) RowIter(ctx *sql.Context, row sql.Row) (sql.RowIter, error) {
    // Phase 1: Accumulate groups
    groups := make(map[string]*AggregateState)
    childIter, _ := g.child.RowIter(ctx, nil)
    
    for {
        inputRow, err := childIter.Next(ctx)
        if err == io.EOF {
            break
        }
        
        // Compute group key
        groupKey := ""
        for _, expr := range g.groupExprs {
            val, _ := expr.Eval(ctx, inputRow)
            groupKey += fmt.Sprintf("%v|", val)
        }
        
        // Initialize or update group state
        if groups[groupKey] == nil {
            groups[groupKey] = &AggregateState{
                groupValues: make([]interface{}, len(g.groupExprs)),
                aggregators: make([]Aggregator, len(g.aggExprs)),
            }
            
            // Store group key values
            for i, expr := range g.groupExprs {
                groups[groupKey].groupValues[i], _ = expr.Eval(ctx, inputRow)
            }
            
            // Initialize aggregators
            for i, aggExpr := range g.aggExprs {
                groups[groupKey].aggregators[i] = createAggregator(aggExpr)
            }
        }
        
        // Update aggregators
        for i, agg := range groups[groupKey].aggregators {
            agg.Update(ctx, inputRow)
        }
    }
    
    // Phase 2: Emit results
    return &groupByIter{groups: groups}, nil
}
```

#### B. Individual Aggregator Implementations

```go
// COUNT(*) aggregator
type CountAggregator struct {
    count int64
}

func (c *CountAggregator) Update(ctx *sql.Context, row sql.Row) {
    c.count++
}

func (c *CountAggregator) Finalize() interface{} {
    return c.count
}

// AVG aggregator
type AvgAggregator struct {
    sum   float64
    count int64
    expr  sql.Expression
}

func (a *AvgAggregator) Update(ctx *sql.Context, row sql.Row) {
    val, err := a.expr.Eval(ctx, row)
    if err == nil && val != nil {
        if num, ok := val.(float64); ok {
            a.sum += num
            a.count++
        }
    }
}

func (a *AvgAggregator) Finalize() interface{} {
    if a.count == 0 {
        return nil
    }
    return a.sum / float64(a.count)
}

// SUM aggregator
type SumAggregator struct {
    sum  float64
    expr sql.Expression
}

func (s *SumAggregator) Update(ctx *sql.Context, row sql.Row) {
    val, err := s.expr.Eval(ctx, row)
    if err == nil && val != nil {
        if num, ok := val.(float64); ok {
            s.sum += num
        }
    }
}

func (s *SumAggregator) Finalize() interface{} {
    return s.sum
}
```

### 3. HAVING Clause Implementation

```go
type Having struct {
    child     sql.Node
    condition sql.Expression
}

func (h *Having) RowIter(ctx *sql.Context, row sql.Row) (sql.RowIter, error) {
    childIter, _ := h.child.RowIter(ctx, nil)
    return &havingIter{
        child:     childIter,
        condition: h.condition,
    }, nil
}

func (iter *havingIter) Next(ctx *sql.Context) (sql.Row, error) {
    for {
        row, err := iter.child.Next(ctx)
        if err != nil {
            return nil, err
        }
        
        // Evaluate HAVING condition on aggregated row
        result, err := iter.condition.Eval(ctx, row)
        if err != nil {
            return nil, err
        }
        
        if result == true {
            return row, nil
        }
        // Continue to next row if condition false
    }
}
```

## Optimization Strategies

### 1. Predicate Pushdown
```go
// Push WHERE conditions down past joins when possible
// Before: JOIN → FILTER
// After:  FILTER → JOIN

// Example: WHERE u.active = 1 AND p.price > 50
// Can push u.active = 1 to users table scan
// Can push p.price > 50 to products table scan
```

### 2. Join Reordering
```go
// Choose optimal join order based on:
// - Table sizes (smaller table as build side in hash join)
// - Selectivity of join conditions
// - Availability of indexes

// Cost-based optimizer evaluates different orders:
// users JOIN products JOIN orders
// vs
// products JOIN orders JOIN users
```

### 3. Index Usage for Joins
```go
// If your storage supports indexes:
type IndexedTable interface {
    sql.Table
    LookupByIndex(ctx *sql.Context, indexName string, key sql.Row) (sql.RowIter, error)
}

// go-mysql-server can use index for join lookups instead of full table scan
```

### 4. Streaming Aggregation
```go
// For sorted input, can compute aggregates without hash table:
// ORDER BY grouping_columns allows streaming aggregation
type StreamingGroupBy struct {
    child      sql.Node
    groupExprs []sql.Expression
    aggExprs   []sql.Expression
}

// Process rows one by one, emit result when group changes
func (s *StreamingGroupBy) processRow(row sql.Row) {
    currentGroupKey := computeGroupKey(row)
    if currentGroupKey != s.lastGroupKey {
        // Emit previous group result
        s.emitGroup()
        // Start new group
        s.startNewGroup(currentGroupKey)
    }
    s.updateCurrentGroup(row)
}
```

This architecture allows go-mysql-server to handle complex SQL queries efficiently while your custom storage layer only needs to implement simple row scanning. The SQL engine handles all the complex join and aggregation logic above your storage interface.