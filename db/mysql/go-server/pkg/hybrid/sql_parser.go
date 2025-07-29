package hybrid

import (
	"fmt"
	"strings"

	"github.com/dolthub/vitess/go/vt/sqlparser"
)

// QueryAnalysis contains the analysis results of a SQL query
type QueryAnalysis struct {
	OriginalQuery     string
	HasCachedTable    bool
	CachedTables      []TableRef
	RemoteTables      []TableRef
	IsJoinQuery       bool
	JoinConditions    []JoinCondition
	SelectColumns     []SelectColumn
	WhereConditions   sqlparser.Expr
	RequiresRewrite   bool
}

// TableRef represents a table reference in the query
type TableRef struct {
	Database string
	Table    string
	Alias    string
}

// JoinCondition represents a join condition between tables
type JoinCondition struct {
	LeftTable   string
	LeftColumn  string
	RightTable  string
	RightColumn string
	Operator    string
}

// SelectColumn represents a selected column
type SelectColumn struct {
	Table  string
	Column string
	Alias  string
	Expr   sqlparser.Expr
}

// SQLParser handles parsing and analyzing SQL queries
type SQLParser struct {
	cachedTables map[string]bool // Map of database.table that are cached
}

// NewSQLParser creates a new SQL parser instance
func NewSQLParser() *SQLParser {
	return &SQLParser{
		cachedTables: make(map[string]bool),
	}
}

// RegisterCachedTable registers a table as being cached in LMDB
func (p *SQLParser) RegisterCachedTable(database, table string) {
	key := fmt.Sprintf("%s.%s", strings.ToLower(database), strings.ToLower(table))
	p.cachedTables[key] = true
}

// UnregisterCachedTable removes a table from the cached list
func (p *SQLParser) UnregisterCachedTable(database, table string) {
	key := fmt.Sprintf("%s.%s", strings.ToLower(database), strings.ToLower(table))
	delete(p.cachedTables, key)
}

// AnalyzeQuery parses and analyzes a SQL query
func (p *SQLParser) AnalyzeQuery(query string, currentDatabase string) (*QueryAnalysis, error) {
	// Parse the SQL query
	stmt, err := sqlparser.Parse(query)
	if err != nil {
		return nil, fmt.Errorf("failed to parse query: %w", err)
	}

	analysis := &QueryAnalysis{
		OriginalQuery:  query,
		CachedTables:   []TableRef{},
		RemoteTables:   []TableRef{},
		SelectColumns:  []SelectColumn{},
		JoinConditions: []JoinCondition{},
	}

	// Handle SELECT statements
	switch stmt := stmt.(type) {
	case *sqlparser.Select:
		return p.analyzeSelect(stmt, currentDatabase, analysis)
	default:
		// For non-SELECT queries, just check if they involve cached tables
		tables := p.extractTables(stmt, currentDatabase)
		for _, table := range tables {
			if p.isTableCached(table.Database, table.Table) {
				analysis.HasCachedTable = true
				analysis.CachedTables = append(analysis.CachedTables, table)
			} else {
				analysis.RemoteTables = append(analysis.RemoteTables, table)
			}
		}
	}

	analysis.RequiresRewrite = analysis.HasCachedTable && len(analysis.RemoteTables) > 0
	return analysis, nil
}

// analyzeSelect analyzes a SELECT statement
func (p *SQLParser) analyzeSelect(stmt *sqlparser.Select, currentDatabase string, analysis *QueryAnalysis) (*QueryAnalysis, error) {
	// Extract selected columns
	for _, sel := range stmt.SelectExprs {
		switch sel := sel.(type) {
		case *sqlparser.AliasedExpr:
			col := p.extractSelectColumn(sel)
			if col != nil {
				analysis.SelectColumns = append(analysis.SelectColumns, *col)
			}
		case *sqlparser.StarExpr:
			// Handle SELECT * case
			tableName := ""
			if sel.TableName.Name.String() != "" {
				tableName = sel.TableName.Name.String()
			}
			analysis.SelectColumns = append(analysis.SelectColumns, SelectColumn{
				Table:  tableName,
				Column: "*",
			})
		}
	}

	// Extract tables from FROM clause
	tables := p.extractTablesFromTableExprs(stmt.From, currentDatabase)
	
	// Classify tables as cached or remote
	for _, table := range tables {
		if p.isTableCached(table.Database, table.Table) {
			analysis.HasCachedTable = true
			analysis.CachedTables = append(analysis.CachedTables, table)
		} else {
			analysis.RemoteTables = append(analysis.RemoteTables, table)
		}
	}

	// Check if it's a join query
	if len(stmt.From) > 1 || p.hasJoinExpr(stmt.From) {
		analysis.IsJoinQuery = true
		// Extract join conditions
		analysis.JoinConditions = p.extractJoinConditions(stmt.From, stmt.Where)
	}

	// Store WHERE conditions
	analysis.WhereConditions = stmt.Where

	analysis.RequiresRewrite = analysis.HasCachedTable && len(analysis.RemoteTables) > 0
	return analysis, nil
}

// extractSelectColumn extracts column information from an aliased expression
func (p *SQLParser) extractSelectColumn(expr *sqlparser.AliasedExpr) *SelectColumn {
	col := &SelectColumn{
		Expr: expr.Expr,
	}

	if expr.As.String() != "" {
		col.Alias = expr.As.String()
	}

	switch e := expr.Expr.(type) {
	case *sqlparser.ColName:
		col.Column = e.Name.String()
		if e.Qualifier.Name.String() != "" {
			col.Table = e.Qualifier.Name.String()
		}
	}

	return col
}

// extractTablesFromTableExprs extracts table references from table expressions
func (p *SQLParser) extractTablesFromTableExprs(tableExprs sqlparser.TableExprs, currentDatabase string) []TableRef {
	var tables []TableRef

	for _, tableExpr := range tableExprs {
		tables = append(tables, p.extractTablesFromTableExpr(tableExpr, currentDatabase)...)
	}

	return tables
}

// extractTablesFromTableExpr extracts table references from a single table expression
func (p *SQLParser) extractTablesFromTableExpr(tableExpr sqlparser.TableExpr, currentDatabase string) []TableRef {
	var tables []TableRef

	switch t := tableExpr.(type) {
	case *sqlparser.AliasedTableExpr:
		switch e := t.Expr.(type) {
		case sqlparser.TableName:
			table := TableRef{
				Database: currentDatabase,
				Table:    e.Name.String(),
			}
			if e.Qualifier.String() != "" {
				table.Database = e.Qualifier.String()
			}
			if t.As.String() != "" {
				table.Alias = t.As.String()
			}
			tables = append(tables, table)
		}
	case *sqlparser.JoinTableExpr:
		// Extract tables from both sides of the join
		tables = append(tables, p.extractTablesFromTableExpr(t.LeftExpr, currentDatabase)...)
		tables = append(tables, p.extractTablesFromTableExpr(t.RightExpr, currentDatabase)...)
	case *sqlparser.ParenTableExpr:
		// Handle parenthesized expressions
		tables = append(tables, p.extractTablesFromTableExprs(t.Exprs, currentDatabase)...)
	}

	return tables
}

// hasJoinExpr checks if table expressions contain joins
func (p *SQLParser) hasJoinExpr(tableExprs sqlparser.TableExprs) bool {
	for _, tableExpr := range tableExprs {
		if _, ok := tableExpr.(*sqlparser.JoinTableExpr); ok {
			return true
		}
	}
	return false
}

// extractJoinConditions extracts join conditions from the query
func (p *SQLParser) extractJoinConditions(tableExprs sqlparser.TableExprs, where *sqlparser.Where) []JoinCondition {
	var conditions []JoinCondition

	// Extract explicit JOIN conditions
	for _, tableExpr := range tableExprs {
		if join, ok := tableExpr.(*sqlparser.JoinTableExpr); ok {
			if join.Condition.On != nil {
				conditions = append(conditions, p.extractConditionsFromExpr(join.Condition.On)...)
			}
		}
	}

	// Extract implicit join conditions from WHERE clause
	if where != nil && where.Expr != nil {
		conditions = append(conditions, p.extractConditionsFromExpr(where.Expr)...)
	}

	return conditions
}

// extractConditionsFromExpr extracts join conditions from an expression
func (p *SQLParser) extractConditionsFromExpr(expr sqlparser.Expr) []JoinCondition {
	var conditions []JoinCondition

	switch e := expr.(type) {
	case *sqlparser.ComparisonExpr:
		if e.Operator == "=" {
			// Check if both sides are column references
			leftCol, leftOk := e.Left.(*sqlparser.ColName)
			rightCol, rightOk := e.Right.(*sqlparser.ColName)
			
			if leftOk && rightOk {
				condition := JoinCondition{
					LeftTable:   leftCol.Qualifier.Name.String(),
					LeftColumn:  leftCol.Name.String(),
					RightTable:  rightCol.Qualifier.Name.String(),
					RightColumn: rightCol.Name.String(),
					Operator:    e.Operator,
				}
				conditions = append(conditions, condition)
			}
		}
	case *sqlparser.AndExpr:
		// Recursively extract from AND expressions
		conditions = append(conditions, p.extractConditionsFromExpr(e.Left)...)
		conditions = append(conditions, p.extractConditionsFromExpr(e.Right)...)
	}

	return conditions
}

// extractTables extracts all table references from any statement type
func (p *SQLParser) extractTables(stmt sqlparser.Statement, currentDatabase string) []TableRef {
	var tables []TableRef

	// Use the Vitess AST walker to find all table references
	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch n := node.(type) {
		case sqlparser.TableName:
			if n.Name.String() != "" {
				table := TableRef{
					Database: currentDatabase,
					Table:    n.Name.String(),
				}
				if n.Qualifier.String() != "" {
					table.Database = n.Qualifier.String()
				}
				tables = append(tables, table)
			}
		}
		return true, nil
	}, stmt)

	return tables
}

// isTableCached checks if a table is cached in LMDB
func (p *SQLParser) isTableCached(database, table string) bool {
	// Special case: always consider ACORDO_GM as cached
	if strings.ToLower(table) == "acordo_gm" {
		return true
	}
	
	key := fmt.Sprintf("%s.%s", strings.ToLower(database), strings.ToLower(table))
	return p.cachedTables[key]
}

// IsCachedTable is a public method to check if a table is cached
func (p *SQLParser) IsCachedTable(database, table string) bool {
	return p.isTableCached(database, table)
}