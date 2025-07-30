package config

import (
	"flag"
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
	"mysql-server-example/pkg/storage"
)

// Config holds all configuration for the MySQL server
type Config struct {
	Server ServerConfig `yaml:"server"`
	Storage StorageConfig `yaml:"storage"`
}

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Port    string `yaml:"port"`
	BindAddr string `yaml:"bind"`
	Debug   bool   `yaml:"debug"`
	Verbose bool   `yaml:"verbose"`
}

// StorageConfig holds storage backend configuration
type StorageConfig struct {
	Backend string `yaml:"backend"` // "mysql", "lmdb", "chdb", "hybrid"
	
	// MySQL passthrough configuration
	MySQL MySQLConfig `yaml:"mysql"`
	
	// LMDB configuration
	LMDB LMDBConfig `yaml:"lmdb"`
	
	// chDB configuration
	ChDB ChDBConfig `yaml:"chdb"`
	
	// Hybrid storage configuration
	Hybrid HybridConfig `yaml:"hybrid"`
}

// MySQLConfig holds MySQL connection configuration
type MySQLConfig struct {
	Host            string `yaml:"host"`
	Port            int    `yaml:"port"`
	User            string `yaml:"user"`
	Password        string `yaml:"password"`
	Database        string `yaml:"database"`
	MaxOpenConns    int    `yaml:"max_open_conns"`
	MaxIdleConns    int    `yaml:"max_idle_conns"`
	ConnMaxLifetime string `yaml:"conn_max_lifetime"`
}

// LMDBConfig holds LMDB configuration
type LMDBConfig struct {
	Path string `yaml:"path"`
}

// HybridConfig contains configuration for hybrid storage routing
type HybridConfig struct {
	HotDataThreshold    int64 `yaml:"hot_data_threshold"`
	AnalyticalThreshold int64 `yaml:"analytical_threshold"`
	AutoMigration       bool  `yaml:"auto_migration"`
	MigrationCheckHours int   `yaml:"migration_check_hours"`
}

// ChDBConfig contains chDB-specific configuration
type ChDBConfig struct {
	Path              string `yaml:"path"`
	MaxMemory         string `yaml:"max_memory"`
	MaxThreads        int    `yaml:"max_threads"`
	Compression       bool   `yaml:"compression"`
	CompressionMethod string `yaml:"compression_method"`
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Port:     "3306",
			BindAddr: "127.0.0.1",
			Debug:    false,
			Verbose:  false,
		},
		Storage: StorageConfig{
			Backend: "lmdb",
			MySQL: MySQLConfig{
				Host:         "localhost",
				Port:         3306,
				User:         "root",
				Password:     "",
				Database:     "",
				MaxOpenConns: 25,
				MaxIdleConns: 5,
				ConnMaxLifetime: "5m",
			},
			LMDB: LMDBConfig{
				Path: "./data",
			},
			ChDB: ChDBConfig{
				Path:              "./chdb_data",
				MaxMemory:         "4G",
				MaxThreads:        4,
				Compression:       true,
				CompressionMethod: "lz4",
			},
			Hybrid: HybridConfig{
				HotDataThreshold:    1_000_000,
				AnalyticalThreshold: 10_000_000,
				AutoMigration:       false,
				MigrationCheckHours: 24,
			},
		},
	}
}

// LoadConfig loads configuration from file, then flags, then environment
func LoadConfig() (*Config, error) {
	// Start with defaults
	cfg := DefaultConfig()
	
	// Try to load from config file
	configFile := os.Getenv("CONFIG_FILE")
	if configFile == "" {
		configFile = "config.yaml"
	}
	
	if data, err := os.ReadFile(configFile); err == nil {
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("failed to parse config file: %w", err)
		}
	}
	
	// Override with command-line flags
	cfg.LoadFromFlags()
	
	// Override with environment variables
	cfg.LoadFromEnv()
	
	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	
	return cfg, nil
}

// LoadFromFlags loads configuration from command-line flags
func (c *Config) LoadFromFlags() {
	// Server flags
	flag.StringVar(&c.Server.Port, "port", c.Server.Port, "Server port")
	flag.StringVar(&c.Server.BindAddr, "bind", c.Server.BindAddr, "Bind address")
	flag.BoolVar(&c.Server.Debug, "debug", c.Server.Debug, "Enable debug mode")
	flag.BoolVar(&c.Server.Verbose, "verbose", c.Server.Verbose, "Enable verbose logging")
	
	// Storage backend
	flag.StringVar(&c.Storage.Backend, "storage", c.Storage.Backend, "Storage backend (mysql, lmdb, chdb, hybrid)")
	
	// MySQL flags
	flag.StringVar(&c.Storage.MySQL.Host, "mysql-host", c.Storage.MySQL.Host, "MySQL host")
	flag.IntVar(&c.Storage.MySQL.Port, "mysql-port", c.Storage.MySQL.Port, "MySQL port")
	flag.StringVar(&c.Storage.MySQL.User, "mysql-user", c.Storage.MySQL.User, "MySQL user")
	flag.StringVar(&c.Storage.MySQL.Password, "mysql-password", c.Storage.MySQL.Password, "MySQL password")
	flag.StringVar(&c.Storage.MySQL.Database, "mysql-database", c.Storage.MySQL.Database, "MySQL database")
	
	// LMDB flags
	flag.StringVar(&c.Storage.LMDB.Path, "lmdb-path", c.Storage.LMDB.Path, "LMDB data directory")
	
	// chDB flags
	flag.StringVar(&c.Storage.ChDB.Path, "chdb-path", c.Storage.ChDB.Path, "chDB data directory")
	flag.StringVar(&c.Storage.ChDB.MaxMemory, "chdb-max-memory", c.Storage.ChDB.MaxMemory, "chDB max memory")
	flag.IntVar(&c.Storage.ChDB.MaxThreads, "chdb-max-threads", c.Storage.ChDB.MaxThreads, "chDB max threads")
	
	// Hybrid flags
	flag.Int64Var(&c.Storage.Hybrid.HotDataThreshold, "hot-data-threshold", c.Storage.Hybrid.HotDataThreshold, "Max rows for LMDB")
	flag.Int64Var(&c.Storage.Hybrid.AnalyticalThreshold, "analytical-threshold", c.Storage.Hybrid.AnalyticalThreshold, "Min rows for chDB")
	
	flag.Parse()
}

// LoadFromEnv loads configuration from environment variables
func (c *Config) LoadFromEnv() {
	// Server settings
	if val := os.Getenv("DEBUG"); val != "" {
		c.Server.Debug = val == "true" || val == "1"
	}
	if val := os.Getenv("VERBOSE"); val != "" {
		c.Server.Verbose = val == "true" || val == "1"
	}
	if val := os.Getenv("PORT"); val != "" {
		c.Server.Port = val
	}
	if val := os.Getenv("BIND_ADDR"); val != "" {
		c.Server.BindAddr = val
	}
	
	// Storage settings
	if val := os.Getenv("STORAGE_BACKEND"); val != "" {
		c.Storage.Backend = val
	}
	
	// MySQL settings
	if val := os.Getenv("MYSQL_HOST"); val != "" {
		c.Storage.MySQL.Host = val
	}
	if val := os.Getenv("MYSQL_PORT"); val != "" {
		if port, err := strconv.Atoi(val); err == nil {
			c.Storage.MySQL.Port = port
		}
	}
	if val := os.Getenv("MYSQL_USER"); val != "" {
		c.Storage.MySQL.User = val
	}
	if val := os.Getenv("MYSQL_PASSWORD"); val != "" {
		c.Storage.MySQL.Password = val
	}
	if val := os.Getenv("MYSQL_DATABASE"); val != "" {
		c.Storage.MySQL.Database = val
	}
	
	// LMDB settings
	if val := os.Getenv("LMDB_PATH"); val != "" {
		c.Storage.LMDB.Path = val
	}
	
	// chDB settings
	if val := os.Getenv("CHDB_PATH"); val != "" {
		c.Storage.ChDB.Path = val
	}
	if val := os.Getenv("CHDB_MAX_MEMORY"); val != "" {
		c.Storage.ChDB.MaxMemory = val
	}
	if val := os.Getenv("CHDB_MAX_THREADS"); val != "" {
		if threads, err := strconv.Atoi(val); err == nil {
			c.Storage.ChDB.MaxThreads = threads
		}
	}
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	// Validate storage backend
	switch c.Storage.Backend {
	case "mysql", "lmdb", "chdb", "hybrid":
		// Valid backends
	default:
		return fmt.Errorf("invalid storage backend: %s", c.Storage.Backend)
	}
	
	// Validate MySQL configuration if using mysql backend
	if c.Storage.Backend == "mysql" {
		if c.Storage.MySQL.Host == "" {
			return fmt.Errorf("mysql host is required")
		}
		if c.Storage.MySQL.Port <= 0 {
			return fmt.Errorf("invalid mysql port: %d", c.Storage.MySQL.Port)
		}
	}
	
	// Validate thresholds
	if c.Storage.Hybrid.HotDataThreshold >= c.Storage.Hybrid.AnalyticalThreshold {
		c.Storage.Hybrid.AnalyticalThreshold = c.Storage.Hybrid.HotDataThreshold * 10
	}
	
	return nil
}

// GetMySQLStorageConfig converts to storage package config
func (c *Config) GetMySQLStorageConfig() storage.MySQLConfig {
	return storage.MySQLConfig{
		Host:            c.Storage.MySQL.Host,
		Port:            c.Storage.MySQL.Port,
		User:            c.Storage.MySQL.User,
		Password:        c.Storage.MySQL.Password,
		Database:        c.Storage.MySQL.Database,
		MaxOpenConns:    c.Storage.MySQL.MaxOpenConns,
		MaxIdleConns:    c.Storage.MySQL.MaxIdleConns,
		ConnMaxLifetime: c.Storage.MySQL.ConnMaxLifetime,
	}
}