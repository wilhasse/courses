package initializer

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/dolthub/go-mysql-server/sql"
	sqle "github.com/dolthub/go-mysql-server"
)

// SQLRunner executes SQL scripts for initialization
type SQLRunner struct {
	engine    *sqle.Engine
	ctx       *sql.Context
}

// NewSQLRunner creates a new SQL script runner
func NewSQLRunner(engine *sqle.Engine) *SQLRunner {
	return &SQLRunner{
		engine: engine,
		ctx:    sql.NewEmptyContext(),
	}
}

// ExecuteScript executes a SQL script file
func (r *SQLRunner) ExecuteScript(scriptPath string) error {
	file, err := os.Open(scriptPath)
	if err != nil {
		return fmt.Errorf("failed to open script file: %w", err)
	}
	defer file.Close()

	var sqlBuffer strings.Builder
	scanner := bufio.NewScanner(file)
	
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		
		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "--") {
			continue
		}
		
		sqlBuffer.WriteString(line)
		sqlBuffer.WriteString(" ")
		
		// Execute when we hit a semicolon
		if strings.HasSuffix(line, ";") {
			statement := strings.TrimSpace(sqlBuffer.String())
			if statement != "" {
				err := r.executeStatement(statement)
				if err != nil {
					return fmt.Errorf("failed to execute statement '%s': %w", statement, err)
				}
			}
			sqlBuffer.Reset()
		}
	}
	
	// Execute any remaining statement
	if sqlBuffer.Len() > 0 {
		statement := strings.TrimSpace(sqlBuffer.String())
		if statement != "" {
			err := r.executeStatement(statement)
			if err != nil {
				return fmt.Errorf("failed to execute final statement '%s': %w", statement, err)
			}
		}
	}
	
	return scanner.Err()
}

// executeStatement executes a single SQL statement
func (r *SQLRunner) executeStatement(statement string) error {
	// Remove trailing semicolon
	statement = strings.TrimSuffix(statement, ";")
	
	// Handle USE database statements to update context
	upperStatement := strings.ToUpper(strings.TrimSpace(statement))
	if strings.HasPrefix(upperStatement, "USE ") {
		dbName := strings.TrimSpace(statement[4:])
		// Clean up database name (remove quotes, semicolons)
		dbName = strings.Trim(dbName, " ;`\"'")
		r.ctx.SetCurrentDatabase(dbName)
	}
	
	_, iter, _, err := r.engine.Query(r.ctx, statement)
	if err != nil {
		// Check if this is a duplicate key error for INSERT statements
		if strings.Contains(strings.ToUpper(statement), "INSERT INTO") && 
		   (strings.Contains(err.Error(), "duplicate") || 
		    strings.Contains(err.Error(), "already exists") ||
		    strings.Contains(err.Error(), "constraint")) {
			// Skip duplicate key errors for INSERT statements during initialization
			fmt.Printf("Skipping duplicate row: %s\n", strings.Split(statement, " VALUES")[0])
			return nil
		}
		return err
	}
	
	// Consume the iterator to ensure the statement is fully executed
	if iter != nil {
		for {
			_, err := iter.Next(r.ctx)
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				return err
			}
		}
		iter.Close(r.ctx)
	}
	
	return nil
}

// CheckInitialized checks if the database has been initialized
func CheckInitialized(engine *sqle.Engine) bool {
	ctx := sql.NewEmptyContext()
	
	// Check if database exists
	_, iter, _, err := engine.Query(ctx, "SHOW DATABASES LIKE 'testdb'")
	if err != nil {
		return false
	}
	if iter != nil {
		row, err := iter.Next(ctx)
		iter.Close(ctx)
		if err != nil || row == nil {
			return false
		}
	} else {
		return false
	}
	
	// Check if tables exist and have data
	_, iter, _, err = engine.Query(ctx, "SELECT COUNT(*) FROM testdb.users")
	if err != nil {
		return false
	}
	
	if iter != nil {
		row, err := iter.Next(ctx)
		iter.Close(ctx)
		if err != nil {
			return false
		}
		
		// Check if users table has data (should have at least 1 row after initialization)
		if row != nil && len(row) > 0 {
			if count, ok := row[0].(int64); ok && count > 0 {
				return true
			}
			if count, ok := row[0].(int32); ok && count > 0 {
				return true
			}
			if count, ok := row[0].(int); ok && count > 0 {
				return true
			}
		}
	}
	
	return false
}