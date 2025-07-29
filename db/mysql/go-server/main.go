package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/dolthub/go-mysql-server/server"
	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/analyzer"
	gms "github.com/dolthub/go-mysql-server"
	"github.com/dolthub/vitess/go/mysql"
	"github.com/rs/zerolog"
	"github.com/sirupsen/logrus"

	"mysql-server-example/pkg/initializer"
	"mysql-server-example/pkg/provider"
	"mysql-server-example/pkg/storage"
)

// DebugTable wraps our table to add execution tracing
type DebugTable struct {
	*provider.Table
	logger *logrus.Logger
}

func NewDebugTable(table *provider.Table, logger *logrus.Logger) *DebugTable {
	return &DebugTable{
		Table:  table,
		logger: logger,
	}
}

func (dt *DebugTable) String() string {
	return fmt.Sprintf("DebugTable(%s)", dt.Table.Name())
}

func (dt *DebugTable) Partitions(ctx *sql.Context) (sql.PartitionIter, error) {
	dt.logger.WithField("table", dt.Name()).Info("🔍 Getting partitions")
	return dt.Table.Partitions(ctx)
}

func (dt *DebugTable) PartitionRows(ctx *sql.Context, partition sql.Partition) (sql.RowIter, error) {
	dt.logger.WithField("table", dt.Name()).Info("📊 Starting table scan")
	iter, err := dt.Table.PartitionRows(ctx, partition)
	if err != nil {
		return nil, err
	}
	return &debugRowIter{
		base:   iter,
		logger: dt.logger,
		table:  dt.Name(),
	}, nil
}

// debugRowIter wraps row iteration to show data flow
type debugRowIter struct {
	base   sql.RowIter
	logger *logrus.Logger
	table  string
	count  int
}

func (d *debugRowIter) Next(ctx *sql.Context) (sql.Row, error) {
	row, err := d.base.Next(ctx)
	if err != nil {
		if err.Error() != "EOF" {
			d.logger.WithFields(logrus.Fields{
				"table": d.table,
				"error": err.Error(),
			}).Error("❌ Error reading row")
		} else {
			d.logger.WithFields(logrus.Fields{
				"table":      d.table,
				"total_rows": d.count,
			}).Info("✅ Finished scanning table")
		}
		return nil, err
	}

	d.count++
	d.logger.WithFields(logrus.Fields{
		"table": d.table,
		"row":   d.count,
		"data":  fmt.Sprintf("%v", row),
	}).Info("📄 Reading row")

	return row, nil
}

func (d *debugRowIter) Close(ctx *sql.Context) error {
	return d.base.Close(ctx)
}

// DebugDatabase wraps database to add tracing
type DebugDatabase struct {
	*provider.Database
	logger *logrus.Logger
}

func NewDebugDatabase(db *provider.Database, logger *logrus.Logger) *DebugDatabase {
	return &DebugDatabase{
		Database: db,
		logger:   logger,
	}
}

func (dd *DebugDatabase) GetTableInsensitive(ctx *sql.Context, tblName string) (sql.Table, bool, error) {
	dd.logger.WithField("table", tblName).Info("🔍 Looking up table")
	
	table, found, err := dd.Database.GetTableInsensitive(ctx, tblName)
	if err != nil {
		dd.logger.WithFields(logrus.Fields{
			"table": tblName,
			"error": err.Error(),
		}).Error("❌ Error getting table")
		return nil, false, err
	}

	if !found {
		dd.logger.WithField("table", tblName).Warn("⚠️ Table not found")
		return nil, false, nil
	}

	dd.logger.WithFields(logrus.Fields{
		"table":   tblName,
		"columns": len(table.Schema()),
	}).Info("✅ Table found")

	// Wrap table in debug wrapper if it's our table type
	if providerTable, ok := table.(*provider.Table); ok {
		return NewDebugTable(providerTable, dd.logger), true, nil
	}

	return table, found, err
}

// DebugDatabaseProvider wraps provider to add tracing
type DebugDatabaseProvider struct {
	sql.DatabaseProvider
	logger *logrus.Logger
}

func NewDebugDatabaseProvider(provider sql.DatabaseProvider, logger *logrus.Logger) *DebugDatabaseProvider {
	return &DebugDatabaseProvider{
		DatabaseProvider: provider,
		logger:           logger,
	}
}

func (ddp *DebugDatabaseProvider) Database(ctx *sql.Context, name string) (sql.Database, error) {
	ddp.logger.WithField("database", name).Info("🔍 Looking up database")

	db, err := ddp.DatabaseProvider.Database(ctx, name)
	if err != nil {
		ddp.logger.WithFields(logrus.Fields{
			"database": name,
			"error":    err.Error(),
		}).Error("❌ Error getting database")
		return nil, err
	}

	ddp.logger.WithField("database", name).Info("✅ Database found")

	// Wrap database in debug wrapper
	if providerDB, ok := db.(*provider.Database); ok {
		return NewDebugDatabase(providerDB, ddp.logger), nil
	}
	
	// Don't wrap RemoteDatabase in debug wrapper for now
	// as it would interfere with proxy functionality
	
	return db, err
}

func main() {
	// Parse command line flags
	debugMode := flag.Bool("debug", false, "Enable debug mode with detailed execution tracing")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	port := flag.String("port", "3306", "Server port")
	bindAddr := flag.String("bind", "127.0.0.1", "Bind address (use 0.0.0.0 for all interfaces)")
	flag.Parse()

	// Check environment variables
	if os.Getenv("DEBUG") == "true" || os.Getenv("DEBUG") == "1" {
		*debugMode = true
	}
	if os.Getenv("VERBOSE") == "true" || os.Getenv("VERBOSE") == "1" {
		*verbose = true
	}
	if envPort := os.Getenv("PORT"); envPort != "" {
		*port = envPort
	}
	if envBind := os.Getenv("BIND_ADDR"); envBind != "" {
		*bindAddr = envBind
	}

	// Set up logging based on flags
	var logger *logrus.Logger
	if *debugMode {
		logger = logrus.New()
		logger.SetLevel(logrus.DebugLevel)
		logger.SetFormatter(&logrus.TextFormatter{
			FullTimestamp: true,
			ForceColors:   true,
		})
	} else if *verbose {
		logrus.SetLevel(logrus.DebugLevel)
		logger = logrus.StandardLogger()
	} else {
		logrus.SetLevel(logrus.InfoLevel)
		logger = logrus.StandardLogger()
	}

	// Create LMDB storage backend
	dbPath := "./data"
	err := os.MkdirAll(dbPath, 0755)
	if err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	// Create zerolog logger for LMDB
	zlogger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	store, err := storage.NewLMDBStorage(dbPath, zlogger)
	if err != nil {
		log.Fatalf("Failed to create LMDB storage: %v", err)
	}
	defer store.Close()

	// Create the database provider with remote database support
	baseProvider := provider.NewDatabaseProvider(store)
	remoteHandler := provider.NewRemoteDatabaseHandler(baseProvider)
	
	var dbProvider sql.DatabaseProvider
	if *debugMode {
		dbProvider = NewDebugDatabaseProvider(remoteHandler, logger)
	} else {
		dbProvider = remoteHandler
	}

	// Create the SQL engine with optional analyzer debugging
	analyzer := analyzer.NewBuilder(dbProvider).Build()
	if *debugMode {
		analyzer.Debug = true
		analyzer.Verbose = true
	}
	
	engine := gms.New(
		analyzer,
		&gms.Config{
			IsReadOnly: false,
		},
	)

	// Check if database needs initialization
	if !initializer.CheckInitialized(engine) {
		logger.Info("Database not initialized. Running initialization script...")
		runner := initializer.NewSQLRunner(engine)
		if err := runner.ExecuteScript("scripts/init.sql"); err != nil {
			log.Fatalf("Failed to initialize database: %v", err)
		}
		logger.Info("Database initialization completed successfully")
	} else {
		logger.Info("Database already initialized")
	}

	// Configure the MySQL server
	address := fmt.Sprintf("%s:%s", *bindAddr, *port)
	config := server.Config{
		Protocol: "tcp",
		Address:  address,
	}

	// Create context factory
	contextFactory := func(ctx context.Context, options ...sql.ContextOption) *sql.Context {
		return sql.NewContext(ctx, options...)
	}

	// Create session factory with optional debug logging
	baseSessionFactory := provider.NewSessionFactory()
	var sessionFactory func(context.Context, *mysql.Conn, string) (sql.Session, error)
	if *debugMode {
		sessionFactory = func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
			session, err := baseSessionFactory(ctx, conn, addr)
			if err != nil {
				return nil, err
			}
			logger.WithFields(logrus.Fields{
				"user":    conn.User,
				"address": addr,
			}).Info("🔗 New session created")
			return session, nil
		}
	} else {
		sessionFactory = baseSessionFactory
	}

	// Create the MySQL server
	s, err := server.NewServer(config, engine, contextFactory, sessionFactory, nil)
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

	if *debugMode {
		logger.Info("🚀 Starting MySQL Server with Debug Mode")
		logger.Info("📋 Sample queries to try:")
		logger.Info("   SELECT * FROM users;")
		logger.Info("   SELECT * FROM products WHERE price > 50;")
		logger.Info("   SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name;")
		logger.Info("   EXPLAIN SELECT * FROM products WHERE price > 100;")
	} else {
		logger.Info("Starting MySQL server")
	}
	
	logger.Infof("Server listening on %s", address)
	if *bindAddr == "0.0.0.0" {
		logger.Infof("Connect with: mysql -h <server-ip> -P %s -u root", *port)
		logger.Warn("Server is accessible from all network interfaces - ensure firewall is properly configured")
	} else {
		logger.Infof("Connect with: mysql -h %s -P %s -u root", *bindAddr, *port)
	}
	
	if *debugMode {
		logger.Info("🔧 Debug mode enabled - detailed execution tracing active")
		logger.Info("💡 To disable debug mode, run without --debug flag or set DEBUG=false")
	}
	
	// Start the server
	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}