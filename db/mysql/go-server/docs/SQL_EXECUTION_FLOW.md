# SQL Execution Flow in go-mysql-server

This document explains how SQL queries are parsed, planned, optimized, and executed in go-mysql-server.

## 1. Overall Architecture Flow

```
SQL Text → Parser → AST → Analyzer → Optimized Plan → Executor → Storage → Results
```

## 2. SQL Parsing Phase

### Input: Raw SQL Text
```sql
SELECT u.name, p.name, p.price 
FROM users u 
JOIN products p ON u.id = p.user_id 
WHERE p.price > 50 
GROUP BY u.id 
HAVING COUNT(*) > 1
```

### Parser (vitess/sqlparser)
go-mysql-server uses Vitess's SQL parser to convert SQL text into an Abstract Syntax Tree (AST):

```go
// Internal flow in go-mysql-server
parsed, err := sqlparser.Parse(query)
// Returns: *sqlparser.Select with structured representation
```

### AST Structure
The parser creates nodes like:
- **SelectStatement**: Main query node
- **TableExprs**: FROM clause (tables, joins)
- **Expr**: WHERE conditions, column references
- **GroupBy**: GROUP BY expressions
- **Having**: HAVING conditions

## 3. Analysis Phase (sql/analyzer)

The analyzer converts the AST into go-mysql-server's internal plan nodes:

### Key Steps:
1. **Resolve References**: Convert table/column names to actual objects
2. **Type Resolution**: Determine data types for expressions
3. **Validation**: Check permissions, schema compatibility
4. **Plan Building**: Create execution plan tree

### Example Plan Tree:
```
Project[u.name, p.name, p.price]
└── Having[COUNT(*) > 1]
    └── GroupBy[u.id]
        └── Filter[p.price > 50]
            └── InnerJoin[u.id = p.user_id]
                ├── TableAlias[u] → ResolvedTable[users]
                └── TableAlias[p] → ResolvedTable[products]
```

## 4. Optimization Phase

The optimizer applies rules to improve performance:

### Common Optimizations:
- **Predicate Pushdown**: Move WHERE conditions closer to tables
- **Projection Pushdown**: Only select needed columns
- **Join Reordering**: Optimize join order based on statistics
- **Index Usage**: Choose optimal indexes

### Example Optimization:
```
// Before optimization
Filter[p.price > 50]
└── InnerJoin[u.id = p.user_id]
    ├── TableScan[users]
    └── TableScan[products]

// After optimization (predicate pushdown)
InnerJoin[u.id = p.user_id]
├── TableScan[users]
└── Filter[p.price > 50]
    └── TableScan[products]
```

## 5. Execution Phase

The executor walks the plan tree and calls your storage interface:

### Row-by-Row Execution
```go
// Simplified execution flow
func (e *Executor) Execute(plan sql.Node) (sql.RowIter, error) {
    switch n := plan.(type) {
    case *plan.ResolvedTable:
        return n.RowIter(ctx, nil) // Calls your storage
    case *plan.Filter:
        childIter, _ := e.Execute(n.Child)
        return &FilterIter{child: childIter, condition: n.Expression}
    case *plan.InnerJoin:
        leftIter, _ := e.Execute(n.Left)
        rightIter, _ := e.Execute(n.Right)
        return &JoinIter{left: leftIter, right: rightIter, condition: n.Cond}
    }
}
```