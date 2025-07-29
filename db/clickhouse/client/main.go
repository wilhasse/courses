package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/ClickHouse/clickhouse-go/v2/lib/driver"
	"gopkg.in/yaml.v3"
)

type Config struct {
	ClickHouse struct {
		Host     string `yaml:"host"`
		Port     int    `yaml:"port"`
		Database string `yaml:"database"`
		Username string `yaml:"username"`
		Password string `yaml:"password"`
		Debug    bool   `yaml:"debug"`
		Settings struct {
			MaxExecutionTime int `yaml:"max_execution_time"`
		} `yaml:"settings"`
		Compression struct {
			Method string `yaml:"method"`
		} `yaml:"compression"`
		Connection struct {
			DialTimeoutSeconds   int `yaml:"dial_timeout_seconds"`
			MaxOpenConns         int `yaml:"max_open_conns"`
			MaxIdleConns         int `yaml:"max_idle_conns"`
			ConnMaxLifetimeHours int `yaml:"conn_max_lifetime_hours"`
			BlockBufferSize      int `yaml:"block_buffer_size"`
		} `yaml:"connection"`
	} `yaml:"clickhouse"`
	Database struct {
		Name  string `yaml:"name"`
		Table string `yaml:"table"`
	} `yaml:"database"`
	Batch struct {
		Size int `yaml:"size"`
	} `yaml:"batch"`
}

func loadConfig(filename string) (*Config, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	return &config, nil
}

func main() {
	ctx := context.Background()
	
	config, err := loadConfig("config.yaml")
	if err != nil {
		log.Fatal("Failed to load config:", err)
	}
	
	conn, err := clickhouse.Open(&clickhouse.Options{
		Addr: []string{fmt.Sprintf("%s:%d", config.ClickHouse.Host, config.ClickHouse.Port)},
		Auth: clickhouse.Auth{
			Database: config.ClickHouse.Database,
			Username: config.ClickHouse.Username,
			Password: config.ClickHouse.Password,
		},
		Debug: config.ClickHouse.Debug,
		Settings: clickhouse.Settings{
			"max_execution_time": config.ClickHouse.Settings.MaxExecutionTime,
		},
		Compression: &clickhouse.Compression{
			Method: clickhouse.CompressionLZ4,
		},
		DialTimeout:          time.Second * time.Duration(config.ClickHouse.Connection.DialTimeoutSeconds),
		MaxOpenConns:         config.ClickHouse.Connection.MaxOpenConns,
		MaxIdleConns:         config.ClickHouse.Connection.MaxIdleConns,
		ConnMaxLifetime:      time.Hour * time.Duration(config.ClickHouse.Connection.ConnMaxLifetimeHours),
		ConnOpenStrategy:     clickhouse.ConnOpenInOrder,
		BlockBufferSize:      uint8(config.ClickHouse.Connection.BlockBufferSize),
	})
	if err != nil {
		log.Fatal("Failed to connect:", err)
	}
	defer conn.Close()

	if err := conn.Ping(ctx); err != nil {
		log.Fatal("Failed to ping:", err)
	}
	
	fmt.Println("Connected to ClickHouse successfully!")

	if err := createDatabase(ctx, conn, config); err != nil {
		log.Fatal("Failed to create database:", err)
	}

	if err := createTable(ctx, conn, config); err != nil {
		log.Fatal("Failed to create table:", err)
	}

	if err := batchInsert(ctx, conn, config); err != nil {
		log.Fatal("Failed to insert data:", err)
	}

	if err := queryData(ctx, conn, config); err != nil {
		log.Fatal("Failed to query data:", err)
	}
}

func createDatabase(ctx context.Context, conn driver.Conn, config *Config) error {
	query := fmt.Sprintf("CREATE DATABASE IF NOT EXISTS %s", config.Database.Name)
	
	if err := conn.Exec(ctx, query); err != nil {
		return fmt.Errorf("create database: %w", err)
	}
	
	fmt.Printf("Database '%s' created successfully!\n", config.Database.Name)
	
	if err := conn.Exec(ctx, fmt.Sprintf("USE %s", config.Database.Name)); err != nil {
		return fmt.Errorf("use database: %w", err)
	}
	
	fmt.Printf("Switched to database '%s'\n", config.Database.Name)
	return nil
}

func createTable(ctx context.Context, conn driver.Conn, config *Config) error {
	query := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			id UInt64,
			name String,
			value Float64,
			created_at DateTime
		) ENGINE = MergeTree()
		ORDER BY (id, created_at)`, config.Database.Name, config.Database.Table)
	
	if err := conn.Exec(ctx, query); err != nil {
		return fmt.Errorf("create table: %w", err)
	}
	
	fmt.Printf("Table '%s' created successfully in %s!\n", config.Database.Table, config.Database.Name)
	return nil
}

func batchInsert(ctx context.Context, conn driver.Conn, config *Config) error {
	batch, err := conn.PrepareBatch(ctx, fmt.Sprintf("INSERT INTO %s.%s (id, name, value, created_at)", config.Database.Name, config.Database.Table))
	if err != nil {
		return fmt.Errorf("prepare batch: %w", err)
	}

	now := time.Now()
	for i := 0; i < config.Batch.Size; i++ {
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

	fmt.Printf("Inserted %d rows successfully!\n", config.Batch.Size)
	return nil
}

func queryData(ctx context.Context, conn driver.Conn, config *Config) error {
	query := fmt.Sprintf(`
		SELECT 
			COUNT(*) as total_rows,
			MIN(value) as min_value,
			MAX(value) as max_value,
			AVG(value) as avg_value
		FROM %s.%s`, config.Database.Name, config.Database.Table)

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

	query = fmt.Sprintf("SELECT id, name, value, created_at FROM %s.%s LIMIT 5", config.Database.Name, config.Database.Table)
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