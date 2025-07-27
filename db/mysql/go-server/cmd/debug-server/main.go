package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/dolthub/go-mysql-server/server"
	"github.com/dolthub/go-mysql-server/sql"
	"github.com/dolthub/go-mysql-server/sql/analyzer"
	"github.com/dolthub/go-mysql-server/sql/types"
	gms "github.com/dolthub/go-mysql-server"
	"github.com/dolthub/vitess/go/mysql"
	"github.com/sirupsen/logrus"

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
	}).Debug("📄 Reading row")

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
	*provider.DatabaseProvider
	logger *logrus.Logger
}

func NewDebugDatabaseProvider(provider *provider.DatabaseProvider, logger *logrus.Logger) *DebugDatabaseProvider {
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

	return db, err
}

func main() {
	// Set up detailed logging
	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
		ForceColors:   true,
	})

	// Create storage with extra data for interesting queries
	store := storage.NewMemoryStorage()
	
	// Create provider with debug wrapper
	baseProvider := provider.NewDatabaseProvider(store)
	debugProvider := NewDebugDatabaseProvider(baseProvider, logger)

	// Create testdb database
	ctx := sql.NewEmptyContext()
	if err := baseProvider.CreateDatabase(ctx, "testdb"); err != nil {
		// It's ok if it already exists
		if !strings.Contains(err.Error(), "database exists") {
			logger.WithError(err).Error("Failed to create testdb")
		}
	}
	
	// Get the database and create sample tables using the built-in method
	db, err := baseProvider.Database(ctx, "testdb")
	if err != nil {
		logger.WithError(err).Error("Failed to get testdb")
	} else if providerDB, ok := db.(*provider.Database); ok {
		providerDB.CreateSampleTables()
		logger.Info("✅ Created sample tables using built-in method")
	}

	// Add more sample data for interesting query results
	addSampleData(store, logger)

	// Create SQL engine with debug analyzer
	analyzer := analyzer.NewBuilder(debugProvider).Build()
	analyzer.Debug = true  // Enable analyzer debugging
	analyzer.Verbose = true

	engine := gms.New(analyzer, &gms.Config{
		IsReadOnly: false,
	})

	// Configure server
	config := server.Config{
		Protocol: "tcp",
		Address:  "127.0.0.1:3311",
	}

	// Create session factory with debug logging
	baseSessionFactory := provider.NewSessionFactory()
	sessionFactory := func(ctx context.Context, conn *mysql.Conn, addr string) (sql.Session, error) {
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

	// Create context factory
	contextFactory := func(ctx context.Context, options ...sql.ContextOption) *sql.Context {
		return sql.NewContext(ctx, options...)
	}

	// Create server
	s, err := server.NewServer(config, engine, contextFactory, sessionFactory, nil)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	// Handle shutdown
	_, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt, syscall.SIGTERM)
		<-c
		logger.Info("🛑 Shutting down debug server...")
		cancel()
		s.Close()
	}()

	logger.Info("🚀 Starting MySQL Debug Server on 127.0.0.1:3311")
	logger.Info("📋 Sample queries to try:")
	logger.Info("   SELECT * FROM users;")
	logger.Info("   SELECT * FROM products WHERE price > 50;")
	logger.Info("   SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name;")
	logger.Info("   EXPLAIN SELECT * FROM products WHERE price > 100;")
	logger.Info("🔌 Connect with: mysql -h 127.0.0.1 -P 3311 -u root")

	if err := s.Start(); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func addSampleData(store *storage.MemoryStorage, logger *logrus.Logger) {
	logger.Info("🔧 Adding extra sample data for demonstrations...")

	// Add more users (tables already created by Database.CreateSampleTables())
	store.InsertRow("testdb", "users", sql.Row{3, "Charlie", "charlie@example.com", "2023-01-03 10:00:00"})
	store.InsertRow("testdb", "users", sql.Row{4, "David", "david@example.com", "2023-01-04 11:00:00"})
	store.InsertRow("testdb", "users", sql.Row{5, "Eve", "eve@example.com", "2023-01-05 12:00:00"})

	// Add more products
	store.InsertRow("testdb", "products", sql.Row{4, "Smartphone", 699.99, "Electronics"})
	store.InsertRow("testdb", "products", sql.Row{5, "Tablet", 399.99, "Electronics"})
	store.InsertRow("testdb", "products", sql.Row{6, "Headphones", 149.99, "Electronics"})
	store.InsertRow("testdb", "products", sql.Row{7, "Monitor", 299.99, "Electronics"})
	store.InsertRow("testdb", "products", sql.Row{8, "Keyboard", 89.99, "Electronics"})

	// Create orders table to demonstrate more complex joins
	ordersSchema := sql.Schema{
		{Name: "id", Type: types.Int32, Nullable: false, PrimaryKey: true},
		{Name: "user_id", Type: types.Int32, Nullable: false},
		{Name: "product_id", Type: types.Int32, Nullable: false},
		{Name: "quantity", Type: types.Int32, Nullable: false},
		{Name: "order_date", Type: types.Timestamp, Nullable: false},
	}

	store.CreateTable("testdb", "orders", ordersSchema)

	// Add sample orders
	store.InsertRow("testdb", "orders", sql.Row{1, 1, 1, 1, "2023-02-01 09:00:00"}) // Alice bought laptop
	store.InsertRow("testdb", "orders", sql.Row{2, 1, 4, 1, "2023-02-01 09:05:00"}) // Alice bought smartphone
	store.InsertRow("testdb", "orders", sql.Row{3, 2, 6, 2, "2023-02-02 14:30:00"}) // Bob bought 2 headphones
	store.InsertRow("testdb", "orders", sql.Row{4, 3, 7, 1, "2023-02-03 16:15:00"}) // Charlie bought monitor
	store.InsertRow("testdb", "orders", sql.Row{5, 3, 8, 1, "2023-02-03 16:20:00"}) // Charlie bought keyboard

	logger.WithFields(logrus.Fields{
		"users":    5,
		"products": 8,
		"orders":   5,
	}).Info("✅ Sample data loaded")

	logger.Info("💡 Try these interesting queries:")
	logger.Info("   -- Users with their order counts:")
	logger.Info("   SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name;")
	logger.Info("")
	logger.Info("   -- Products ordered with user info:")
	logger.Info("   SELECT u.name, p.name, o.quantity, o.order_date FROM users u JOIN orders o ON u.id = o.user_id JOIN products p ON o.product_id = p.id ORDER BY o.order_date;")
	logger.Info("")
	logger.Info("   -- High-value orders (products > $200):")
	logger.Info("   SELECT u.name, p.name, p.price FROM users u JOIN orders o ON u.id = o.user_id JOIN products p ON o.product_id = p.id WHERE p.price > 200;")
}