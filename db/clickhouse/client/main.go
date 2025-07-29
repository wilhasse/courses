package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
)

func main() {
	ctx := context.Background()
	
	conn, err := clickhouse.Open(&clickhouse.Options{
		Addr: []string{"192.168.20.16:9000"},
		Auth: clickhouse.Auth{
			Database: "default",
			Username: "root",
			Password: "test123",
		},
		Debug: true,
		Settings: clickhouse.Settings{
			"max_execution_time": 60,
		},
		Compression: &clickhouse.Compression{
			Method: clickhouse.CompressionLZ4,
		},
		DialTimeout:          time.Second * 30,
		MaxOpenConns:         5,
		MaxIdleConns:         5,
		ConnMaxLifetime:      time.Hour,
		ConnOpenStrategy:     clickhouse.ConnOpenInOrder,
		BlockBufferSize:      10,
	})
	if err != nil {
		log.Fatal("Failed to connect:", err)
	}
	defer conn.Close()

	if err := conn.Ping(ctx); err != nil {
		log.Fatal("Failed to ping:", err)
	}
	
	fmt.Println("Connected to ClickHouse successfully!")

	if err := createDatabase(ctx, conn); err != nil {
		log.Fatal("Failed to create database:", err)
	}

	if err := createTable(ctx, conn); err != nil {
		log.Fatal("Failed to create table:", err)
	}

	if err := batchInsert(ctx, conn); err != nil {
		log.Fatal("Failed to insert data:", err)
	}

	if err := queryData(ctx, conn); err != nil {
		log.Fatal("Failed to query data:", err)
	}
}

func createDatabase(ctx context.Context, conn driver.Conn) error {
	query := `CREATE DATABASE IF NOT EXISTS test_db`
	
	if err := conn.Exec(ctx, query); err != nil {
		return fmt.Errorf("create database: %w", err)
	}
	
	fmt.Println("Database 'test_db' created successfully!")
	
	if err := conn.Exec(ctx, "USE test_db"); err != nil {
		return fmt.Errorf("use database: %w", err)
	}
	
	fmt.Println("Switched to database 'test_db'")
	return nil
}

func createTable(ctx context.Context, conn driver.Conn) error {
	query := `
		CREATE TABLE IF NOT EXISTS test_db.example_batch (
			id UInt64,
			name String,
			value Float64,
			created_at DateTime
		) ENGINE = MergeTree()
		ORDER BY (id, created_at)
	`
	
	if err := conn.Exec(ctx, query); err != nil {
		return fmt.Errorf("create table: %w", err)
	}
	
	fmt.Println("Table created successfully in test_db!")
	return nil
}

func batchInsert(ctx context.Context, conn driver.Conn) error {
	batch, err := conn.PrepareBatch(ctx, "INSERT INTO test_db.example_batch (id, name, value, created_at)")
	if err != nil {
		return fmt.Errorf("prepare batch: %w", err)
	}

	now := time.Now()
	for i := 0; i < 1000; i++ {
		err := batch.Append(
			uint64(i),
			fmt.Sprintf("item_%d", i),
			float64(i) * 1.23,
			now.Add(time.Duration(i) * time.Second),
		)
		if err != nil {
			return fmt.Errorf("append to batch: %w", err)
		}
	}

	if err := batch.Send(); err != nil {
		return fmt.Errorf("send batch: %w", err)
	}

	fmt.Println("Inserted 1000 rows successfully!")
	return nil
}

func queryData(ctx context.Context, conn driver.Conn) error {
	query := `
		SELECT 
			COUNT(*) as total_rows,
			MIN(value) as min_value,
			MAX(value) as max_value,
			AVG(value) as avg_value
		FROM test_db.example_batch
	`

	row := conn.QueryRow(ctx, query)
	
	var totalRows uint64
	var minValue, maxValue, avgValue float64
	
	if err := row.Scan(&totalRows, &minValue, &maxValue, &avgValue); err != nil {
		return fmt.Errorf("scan row: %w", err)
	}

	fmt.Printf("\nQuery Results:\n")
	fmt.Printf("Total Rows: %d\n", totalRows)
	fmt.Printf("Min Value: %.2f\n", minValue)
	fmt.Printf("Max Value: %.2f\n", maxValue)
	fmt.Printf("Average Value: %.2f\n", avgValue)

	query = "SELECT id, name, value, created_at FROM test_db.example_batch LIMIT 5"
	rows, err := conn.Query(ctx, query)
	if err != nil {
		return fmt.Errorf("query rows: %w", err)
	}
	defer rows.Close()

	fmt.Println("\nFirst 5 rows:")
	fmt.Println("ID\tName\t\tValue\t\tCreated At")
	fmt.Println("--\t----\t\t-----\t\t----------")
	
	for rows.Next() {
		var (
			id        uint64
			name      string
			value     float64
			createdAt time.Time
		)
		if err := rows.Scan(&id, &name, &value, &createdAt); err != nil {
			return fmt.Errorf("scan rows: %w", err)
		}
		fmt.Printf("%d\t%s\t%.2f\t\t%s\n", id, name, value, createdAt.Format("2006-01-02 15:04:05"))
	}

	return rows.Err()
}