package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/dolthub/go-mysql-server/server"
	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/analyzer"
	gms "github.com/dolthub/go-mysql-server"
	"github.com/sirupsen/logrus"

	"mysql-server-example/pkg/provider"
	"mysql-server-example/pkg/storage"
)

func main() {
	// Set up logging
	logrus.SetLevel(logrus.InfoLevel)
	logger := logrus.StandardLogger()

	// Create our custom storage backend
	store := storage.NewMemoryStorage()

	// Create the database provider
	dbProvider := provider.NewDatabaseProvider(store)

	// Create testdb database with sample data
	ctx := sql.NewEmptyContext()
	if err := dbProvider.CreateDatabase(ctx, "testdb"); err != nil {
		logger.WithError(err).Warn("Failed to create testdb (may already exist)")
	}
	
	// Get the database and create sample tables
	db, err := dbProvider.Database(ctx, "testdb")
	if err == nil {
		if providerDB, ok := db.(*provider.Database); ok {
			providerDB.CreateSampleTables()
			logger.Info("Created sample tables in testdb")
		}
	}

	// Create the SQL engine
	engine := gms.New(
		analyzer.NewBuilder(dbProvider).Build(),
		&gms.Config{
			IsReadOnly: false,
		},
	)

	// Configure the MySQL server
	config := server.Config{
		Protocol: "tcp",
		Address:  "127.0.0.1:3306",
	}

	// Create context factory
	contextFactory := func(ctx context.Context, options ...sql.ContextOption) *sql.Context {
		return sql.NewContext(ctx, options...)
	}

	// Create the MySQL server
	s, err := server.NewServer(config, engine, contextFactory, provider.NewSessionFactory(), nil)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Handle graceful shutdown
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		logger.Info("Shutting down server...")
		cancel()
		s.Close()
	}()

	logger.Info("Starting MySQL server on 127.0.0.1:3306")
	logger.Info("Connect with: mysql -h 127.0.0.1 -P 3306 -u root")
	
	// Start the server
	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}