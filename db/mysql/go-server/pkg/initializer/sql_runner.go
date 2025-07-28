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
			return nil
		}
		
		// Check if this is a table already exists error for CREATE TABLE statements
		if strings.Contains(strings.ToUpper(statement), "CREATE TABLE") && 
		   strings.Contains(err.Error(), "already exists") {
			// Skip table already exists errors during initialization
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
	
	// Check if the testdb database exists and has tables
	_, iter, _, err := engine.Query(ctx, "SHOW DATABASES")
	if err != nil {
		return false
	}
	
	// Look for testdb in the databases list
	hasTestDB := false
	if iter != nil {
		for {
			row, err := iter.Next(ctx)
			if err != nil {
				if err.Error() == "EOF" {
					break
				}
				iter.Close(ctx)
				return false
			}
			if row != nil && len(row) > 0 {
				if dbName, ok := row[0].(string); ok && dbName == "testdb" {
					hasTestDB = true
					break
				}
			}
		}
		iter.Close(ctx)
	}
	
	if !hasTestDB {
		return false
	}
	
	// If testdb exists, check if it has the users table with data
	_, iter, _, err = engine.Query(ctx, "SELECT COUNT(*) FROM testdb.users")
	if err != nil {
		// Table doesn't exist or can't be queried
		return false
	}
	
	if iter == nil {
		return false
	}
	
	row, err := iter.Next(ctx)
	iter.Close(ctx)
	if err != nil {
		return false
	}
	
	if row == nil || len(row) == 0 {
		return false
	}
	
	// Check if we have data (count > 0)
	switch v := row[0].(type) {
	case int64:
		return v > 0
	case int32:
		return v > 0
	case int:
		return v > 0
	case uint64:
		return v > 0
	case uint32:
		return v > 0
	case uint:
		return v > 0
	default:
		return false
	}
}