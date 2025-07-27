package provider

import (
	"context"

	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/vitess/go/mysql"
)

// NewSessionFactory creates a session builder function
func NewSessionFactory() func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
	return func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
		// Create a base session
		session := sql.NewBaseSession()
		
		// Set the client information
		session.SetClient(sql.Client{
			User:         conn.User,
			Address:      addr,
			Capabilities: 0,
		})

		// Set default database if desired
		// session.SetCurrentDatabase("testdb")

		return session, nil
	}
}