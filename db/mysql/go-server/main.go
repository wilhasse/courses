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
	"github.com/dolthub/vitess/go/mysql"
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
		Address:  "localhost:3306",
	}

	// Create the MySQL server
	s, err := server.NewServer(config, engine, provider.NewSessionFactory(), provider.NewSessionFactory(), nil)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Handle graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		logger.Info("Shutting down server...")
		cancel()
		s.Close()
	}()

	logger.Info("Starting MySQL server on localhost:3306")
	logger.Info("Connect with: mysql -h localhost -P 3306 -u root")
	
	// Start the server
	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}