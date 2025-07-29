package hybrid

import (
	"fmt"

	"github.com/dolthub/vitess/go/vt/sqlparser"
)

// RewriteResult contains the result of query rewriting
type RewriteResult struct {
	RemoteQuery      string              // Query to execute on remote MySQL (without cached tables)
	CachedTableNames []string            // Names of cached tables to query from LMDB
	JoinStrategy     JoinStrategy        // How to join the results
	OriginalQuery    string              // Original query for reference
}

// JoinStrategy defines how to join results from different sources
type JoinStrategy struct {
	Type       string // "nested_loop", "hash", "merge"
	Conditions []JoinCondition
}

// QueryRewriter handles rewriting queries to split between MySQL and LMDB
type QueryRewriter struct {
	parser *SQLParser
}

// NewQueryRewriter creates a new query rewriter instance
func NewQueryRewriter(parser *SQLParser) *QueryRewriter {
	return &QueryRewriter{
		parser: parser,
	}
}

// RewriteQuery rewrites a query to split execution between MySQL and LMDB
func (r *QueryRewriter) RewriteQuery(query string, currentDatabase string) (*RewriteResult, error) {
	// First analyze the query
	analysis, err := r.parser.AnalyzeQuery(query, currentDatabase)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query: %w", err)
	}

	// If no rewrite is needed, return the original query
	if !analysis.RequiresRewrite {
		return &RewriteResult{
			RemoteQuery:   query,
			OriginalQuery: query,
		}, nil
	}

	// Parse the query for rewriting
	stmt, err := sqlparser.Parse(query)
	if err != nil {
		return nil, fmt.Errorf("failed to parse query for rewriting: %w", err)
	}

	// Handle SELECT statements
	selectStmt, ok := stmt.(*sqlparser.Select)
	if !ok {
		return nil, fmt.Errorf("query rewriting is currently only supported for SELECT statements")
	}

	// Create the rewrite result
	result := &RewriteResult{
		OriginalQuery:    query,
		CachedTableNames: make([]string, 0),
	}

	// Extract cached table names
	for _, table := range analysis.CachedTables {
		tableName := table.Table
		if table.Database != "" && table.Database != currentDatabase {
			tableName = fmt.Sprintf("%s.%s", table.Database, table.Table)
		}
		result.CachedTableNames = append(result.CachedTableNames, tableName)
	}

	// Rewrite the query to remove cached tables
	rewrittenStmt := r.rewriteSelectStatement(selectStmt, analysis, currentDatabase)

	// Generate the rewritten query string
	buf := sqlparser.NewTrackedBuffer(nil)
	rewrittenStmt.Format(buf)
	result.RemoteQuery = buf.String()

	// Set join strategy if needed
	if analysis.IsJoinQuery {
		result.JoinStrategy = JoinStrategy{
			Type:       "nested_loop", // Default to nested loop join
			Conditions: analysis.JoinConditions,
		}
	}

	return result, nil
}

// rewriteSelectStatement rewrites a SELECT statement to remove cached tables
func (r *QueryRewriter) rewriteSelectStatement(stmt *sqlparser.Select, analysis *QueryAnalysis, currentDatabase string) *sqlparser.Select {
	// Clone the statement to avoid modifying the original
	rewritten := &sqlparser.Select{
		SelectExprs: r.rewriteSelectExprs(stmt.SelectExprs, analysis),
		From:        r.rewriteTableExprs(stmt.From, analysis, currentDatabase),
		Where:       r.rewriteWhere(stmt.Where, analysis),
		GroupBy:     stmt.GroupBy,
		Having:      stmt.Having,
		OrderBy:     stmt.OrderBy,
		Limit:       stmt.Limit,
	}

	return rewritten
}

// rewriteSelectExprs rewrites SELECT expressions to remove references to cached tables
func (r *QueryRewriter) rewriteSelectExprs(exprs sqlparser.SelectExprs, analysis *QueryAnalysis) sqlparser.SelectExprs {
	var rewritten sqlparser.SelectExprs
	var neededJoinColumns []string

	// First, identify join columns we need from remote tables
	for _, condition := range analysis.JoinConditions {
		// Check if the right side is from a remote table
		for _, remoteTable := range analysis.RemoteTables {
			if condition.RightTable == remoteTable.Table || condition.RightTable == remoteTable.Alias {
				neededJoinColumns = append(neededJoinColumns, condition.RightColumn)
			}
		}
	}

	for _, expr := range exprs {
		switch e := expr.(type) {
		case *sqlparser.AliasedExpr:
			// Check if this column references a cached table
			if col, ok := e.Expr.(*sqlparser.ColName); ok {
				if r.isColumnFromCachedTable(col, analysis) {
					// Skip columns from cached tables in remote query
					continue
				}
			}
			rewritten = append(rewritten, expr)
		case *sqlparser.StarExpr:
			// Handle SELECT * case
			if e.TableName.Name.String() == "" {
				// SELECT * without table qualifier - need to expand
				// For now, keep it as is (this would need more complex handling)
				rewritten = append(rewritten, expr)
			} else {
				// Check if the table is cached
				tableName := e.TableName.Name.String()
				if !r.isTableCached(tableName, analysis) {
					rewritten = append(rewritten, expr)
				}
			}
		}
	}

	// Add join columns if they're not already included
	for _, colName := range neededJoinColumns {
		// Check if this column is already in the rewritten list
		found := false
		for _, expr := range rewritten {
			if aliased, ok := expr.(*sqlparser.AliasedExpr); ok {
				if col, ok := aliased.Expr.(*sqlparser.ColName); ok {
					if col.Name.String() == colName {
						found = true
						break
					}
				}
			}
		}
		
		if !found {
			// Add the join column from the remote table
			for _, remoteTable := range analysis.RemoteTables {
				for _, condition := range analysis.JoinConditions {
					if condition.RightColumn == colName && 
					   (condition.RightTable == remoteTable.Table || condition.RightTable == remoteTable.Alias) {
						// Create a new column expression for the join column
						newCol := &sqlparser.AliasedExpr{
							Expr: &sqlparser.ColName{
								Name: sqlparser.NewColIdent(colName),
								Qualifier: sqlparser.TableName{
									Name: sqlparser.NewTableIdent(remoteTable.Alias),
								},
							},
						}
						rewritten = append(rewritten, newCol)
						break
					}
				}
			}
		}
	}

	// If all columns were removed, add a dummy column to make the query valid
	if len(rewritten) == 0 {
		rewritten = append(rewritten, &sqlparser.AliasedExpr{
			Expr: &sqlparser.SQLVal{Type: sqlparser.IntVal, Val: []byte("1")},
			As: sqlparser.NewColIdent("dummy"),
		})
	}

	return rewritten
}

// rewriteTableExprs rewrites FROM clause to remove cached tables
func (r *QueryRewriter) rewriteTableExprs(exprs sqlparser.TableExprs, analysis *QueryAnalysis, currentDatabase string) sqlparser.TableExprs {
	var rewritten sqlparser.TableExprs

	for _, expr := range exprs {
		rewrittenExpr := r.rewriteTableExpr(expr, analysis, currentDatabase)
		if rewrittenExpr != nil {
			rewritten = append(rewritten, rewrittenExpr)
		}
	}

	return rewritten
}

// rewriteTableExpr rewrites a single table expression
func (r *QueryRewriter) rewriteTableExpr(expr sqlparser.TableExpr, analysis *QueryAnalysis, currentDatabase string) sqlparser.TableExpr {
	switch t := expr.(type) {
	case *sqlparser.AliasedTableExpr:
		if tableName, ok := t.Expr.(sqlparser.TableName); ok {
			table := tableName.Name.String()
			database := currentDatabase
			if tableName.DbQualifier.String() != "" {
				database = tableName.DbQualifier.String()
			}
			
			// Check if this table is cached
			for _, cached := range analysis.CachedTables {
				if cached.Table == table && cached.Database == database {
					// This table is cached, remove it from remote query
					return nil
				}
			}
		}
		return expr
	case *sqlparser.JoinTableExpr:
		// Rewrite both sides of the join
		left := r.rewriteTableExpr(t.LeftExpr, analysis, currentDatabase)
		right := r.rewriteTableExpr(t.RightExpr, analysis, currentDatabase)

		// If both sides are removed, return nil
		if left == nil && right == nil {
			return nil
		}

		// If only one side remains, return that side without the join
		if left == nil {
			return right
		}
		if right == nil {
			return left
		}

		// Both sides remain, keep the join
		return &sqlparser.JoinTableExpr{
			LeftExpr:  left,
			Join:      t.Join,
			RightExpr: right,
			Condition: r.rewriteJoinCondition(&t.Condition, analysis),
		}
	case *sqlparser.ParenTableExpr:
		rewritten := r.rewriteTableExprs(t.Exprs, analysis, currentDatabase)
		if len(rewritten) == 0 {
			return nil
		}
		return &sqlparser.ParenTableExpr{Exprs: rewritten}
	}

	return expr
}

// rewriteJoinCondition rewrites join conditions to remove references to cached tables
func (r *QueryRewriter) rewriteJoinCondition(cond *sqlparser.JoinCondition, analysis *QueryAnalysis) sqlparser.JoinCondition {
	if cond == nil || cond.On == nil {
		return sqlparser.JoinCondition{}
	}

	// For now, we'll keep the join condition as is
	// In a more sophisticated implementation, we would filter out conditions
	// that only involve cached tables
	return *cond
}

// rewriteWhere rewrites WHERE clause to remove conditions on cached tables
func (r *QueryRewriter) rewriteWhere(where *sqlparser.Where, analysis *QueryAnalysis) *sqlparser.Where {
	if where == nil || where.Expr == nil {
		return where
	}

	// Rewrite the WHERE expression
	rewrittenExpr := r.rewriteExpr(where.Expr, analysis)
	if rewrittenExpr == nil {
		return nil
	}

	return &sqlparser.Where{
		Type: where.Type,
		Expr: rewrittenExpr,
	}
}

// rewriteExpr rewrites an expression to remove references to cached tables
func (r *QueryRewriter) rewriteExpr(expr sqlparser.Expr, analysis *QueryAnalysis) sqlparser.Expr {
	switch e := expr.(type) {
	case *sqlparser.ComparisonExpr:
		// Check if either side references a cached table
		leftHasCached := r.exprReferencesCachedTable(e.Left, analysis)
		rightHasCached := r.exprReferencesCachedTable(e.Right, analysis)

		if leftHasCached && rightHasCached {
			// Both sides reference cached tables, remove this condition
			return nil
		}

		if !leftHasCached && !rightHasCached {
			// Neither side references cached tables, keep the condition
			return expr
		}

		// One side references cached table, this is a join condition
		// Keep it for now (will be used in join logic)
		return expr

	case *sqlparser.AndExpr:
		left := r.rewriteExpr(e.Left, analysis)
		right := r.rewriteExpr(e.Right, analysis)

		if left == nil && right == nil {
			return nil
		}
		if left == nil {
			return right
		}
		if right == nil {
			return left
		}
		return &sqlparser.AndExpr{Left: left, Right: right}

	case *sqlparser.OrExpr:
		left := r.rewriteExpr(e.Left, analysis)
		right := r.rewriteExpr(e.Right, analysis)

		if left == nil && right == nil {
			return nil
		}
		if left == nil {
			return right
		}
		if right == nil {
			return left
		}
		return &sqlparser.OrExpr{Left: left, Right: right}

	default:
		// For other expression types, check if they reference cached tables
		if r.exprReferencesCachedTable(expr, analysis) {
			return nil
		}
		return expr
	}
}

// exprReferencesCachedTable checks if an expression references a cached table
func (r *QueryRewriter) exprReferencesCachedTable(expr sqlparser.Expr, analysis *QueryAnalysis) bool {
	hasReference := false

	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		if col, ok := node.(*sqlparser.ColName); ok {
			if r.isColumnFromCachedTable(col, analysis) {
				hasReference = true
				return false, nil
			}
		}
		return true, nil
	}, expr)

	return hasReference
}

// isColumnFromCachedTable checks if a column reference is from a cached table
func (r *QueryRewriter) isColumnFromCachedTable(col *sqlparser.ColName, analysis *QueryAnalysis) bool {
	tableName := col.Qualifier.Name.String()
	if tableName == "" {
		// No table qualifier, need to check all cached tables
		// This is ambiguous, for safety assume it could be from cached table
		return len(analysis.CachedTables) > 0
	}

	// Check if the table name matches any cached table
	for _, cached := range analysis.CachedTables {
		if cached.Table == tableName || cached.Alias == tableName {
			return true
		}
	}

	return false
}

// isTableCached checks if a table name refers to a cached table
func (r *QueryRewriter) isTableCached(tableName string, analysis *QueryAnalysis) bool {
	for _, cached := range analysis.CachedTables {
		if cached.Table == tableName || cached.Alias == tableName {
			return true
		}
	}
	return false
}