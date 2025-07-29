package provider

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/dolthub/go-mysql-server/sql"
)

// RemoteDatabaseHandler wraps DatabaseProvider to intercept and handle special CREATE DATABASE syntax
type RemoteDatabaseHandler struct {
	*DatabaseProvider
}

// NewRemoteDatabaseHandler creates a new handler that supports remote database syntax
func NewRemoteDatabaseHandler(provider *DatabaseProvider) *RemoteDatabaseHandler {
	return &RemoteDatabaseHandler{
		DatabaseProvider: provider,
	}
}

// CreateDatabase overrides to support remote database creation syntax
// Syntax: CREATE DATABASE dbname REMOTE 'host:port/database@user:password'
// Example: CREATE DATABASE myproxy REMOTE 'localhost:3306/production@root:password'
func (h *RemoteDatabaseHandler) CreateDatabase(ctx *sql.Context, name string) error {
	// Check if this is a remote database creation request
	// Since go-mysql-server's parser doesn't support custom syntax, we'll use a naming convention
	// Format: dbname__remote__host__port__database__user__password
	if strings.Contains(name, "__remote__") {
		config, parsedName, err := parseRemoteDatabaseName(name)
		if err != nil {
			return err
		}
		return h.DatabaseProvider.CreateRemoteDatabase(ctx, parsedName, config)
	}

	// Regular database creation
	return h.DatabaseProvider.CreateDatabase(ctx, name)
}

// parseRemoteDatabaseName parses the special naming convention for remote databases
// Format: dbname__remote__host__port__database__user__password
func parseRemoteDatabaseName(name string) (RemoteConfig, string, error) {
	parts := strings.Split(name, "__")
	if len(parts) < 7 || parts[1] != "remote" {
		return RemoteConfig{}, "", fmt.Errorf("invalid remote database format")
	}

	dbName := parts[0]
	// Convert underscores back to dots for IP addresses
	host := strings.ReplaceAll(parts[2], "_", ".")
	port, err := strconv.Atoi(parts[3])
	if err != nil {
		return RemoteConfig{}, "", fmt.Errorf("invalid port: %v", err)
	}
	
	// Handle special characters in password
	password := parts[6]
	// Replace AT with @ in password
	password = strings.ReplaceAll(password, "AT", "@")
	
	config := RemoteConfig{
		Host:     host,
		Port:     port,
		Database: parts[4],
		User:     parts[5],
		Password: password,
	}

	return config, dbName, nil
}

// ParseRemoteDatabaseCommand provides an alternative way to parse remote database commands
// This can be used with a custom SQL function or comment-based syntax
func ParseRemoteDatabaseCommand(query string) (string, RemoteConfig, bool) {
	// Pattern 1: CREATE DATABASE name COMMENT 'remote:host:port/database@user:password'
	commentPattern := regexp.MustCompile(`(?i)CREATE\s+DATABASE\s+(\w+)\s+COMMENT\s+'remote:([^:]+):(\d+)/([^@]+)@([^:]+):(.+)'`)
	if matches := commentPattern.FindStringSubmatch(query); matches != nil {
		port, _ := strconv.Atoi(matches[3])
		config := RemoteConfig{
			Host:     matches[2],
			Port:     port,
			Database: matches[4],
			User:     matches[5],
			Password: matches[6],
		}
		return matches[1], config, true
	}

	// Pattern 2: Using special function syntax (for future enhancement)
	// CREATE DATABASE name AS REMOTE_CONNECTION('host:port', 'database', 'user', 'password')
	
	return "", RemoteConfig{}, false
}

// CreateRemoteDatabaseHelper creates a remote database using a more user-friendly syntax
// This can be called from a stored procedure or custom function
func (h *RemoteDatabaseHandler) CreateRemoteDatabaseHelper(ctx *sql.Context, name, host string, port int, database, user, password string) error {
	config := RemoteConfig{
		Host:     host,
		Port:     port,
		Database: database,
		User:     user,
		Password: password,
	}
	return h.DatabaseProvider.CreateRemoteDatabase(ctx, name, config)
}