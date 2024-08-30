package main

import (
	"fmt"
	"strings"

        "github.com/pingcap/tidb/pkg/parser"
        "github.com/pingcap/tidb/pkg/parser/ast"
        "github.com/pingcap/tidb/pkg/parser/mysql"
        "github.com/pingcap/tidb/pkg/parser/types"
        _ "github.com/pingcap/tidb/pkg/parser/test_driver"
)

func main() {
	// Example usage
	mysqlStmt := `CREATE TABLE users (
		id INT PRIMARY KEY,
		name VARCHAR(255),
		age INT,
		created_at TIMESTAMP
	) ENGINE=InnoDB;`

	dorisStmt, err := convertMySQLToDoris(mysqlStmt)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Println(dorisStmt)
}

func convertMySQLToDoris(mysqlStmt string) (string, error) {
	p := parser.New()
	stmt, err := p.ParseOneStmt(mysqlStmt, "", "")
	if err != nil {
		return "", fmt.Errorf("parse error: %v", err)
	}

	createStmt, ok := stmt.(*ast.CreateTableStmt)
	if !ok {
		return "", fmt.Errorf("not a CREATE TABLE statement")
	}

	var sb strings.Builder
	sb.WriteString("CREATE TABLE ")
	sb.WriteString(createStmt.Table.Name.String())
	sb.WriteString(" (\n")

	for i, col := range createStmt.Cols {
		if i > 0 {
			sb.WriteString(",\n")
		}
		sb.WriteString("  ")
		sb.WriteString(col.Name.Name.O)
		sb.WriteString(" ")
		sb.WriteString(convertDataType(col.Tp))
	}

	sb.WriteString("\n) ENGINE=OLAP")

	// TODO: Handle keys, partitions, and other table options

	return sb.String(), nil
}

func convertDataType(tp *types.FieldType) string {
	switch tp.GetType() {
	case mysql.TypeTiny, mysql.TypeShort, mysql.TypeLong, mysql.TypeLonglong:
		return "INT"
	case mysql.TypeFloat, mysql.TypeDouble:
		return "DOUBLE"
	case mysql.TypeNewDecimal:
		return fmt.Sprintf("DECIMAL(%d,%d)", tp.GetFlen(), tp.GetDecimal())
	case mysql.TypeVarchar, mysql.TypeVarString:
		return fmt.Sprintf("VARCHAR(%d)", tp.GetFlen())
	case mysql.TypeTimestamp:
		return "DATETIME"
	// Add more type conversions as needed
	default:
		return "STRING" // Default to STRING for unsupported types
	}
}
