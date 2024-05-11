package main

import (
	"fmt"

	"vitess.io/vitess/go/vt/sqlparser"
)

func main() {
	sql := "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders WHERE total > 100);"

	// Parse the SQL into an AST
	stmt, err := sqlparser.Parse(sql)
	if err != nil {
		fmt.Printf("Error parsing SQL: %v\n", err)
		return
	}

	// Type assert to a Select statement
	selectStmt, ok := stmt.(*sqlparser.Select)
	if !ok {
		fmt.Println("Not a SELECT statement.")
		return
	}

	// Initialize an empty string for the subquery
	var subquerySQL string

	// Traverse the AST
	sqlparser.Walk(func(node sqlparser.SQLNode) (kontinue bool, err error) {
		switch node := node.(type) {
		case *sqlparser.Subquery:
			// Serialize the subquery to SQL
			subquerySQL = sqlparser.String(node)
		}
		return true, nil
	}, selectStmt)

	// Serialize the main query (with subquery replaced to make it standalone if necessary)
	mainQuerySQL := sqlparser.String(selectStmt)

	// Print the results
	fmt.Println("Main Query:", mainQuerySQL)
	fmt.Println("Subquery:", subquerySQL)
}
