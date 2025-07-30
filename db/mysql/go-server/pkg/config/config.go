package config

import (
	"flag"
	"os"
	"strconv"
)

// Config holds all configuration for the MySQL server
type Config struct {
	// Server configuration
	Debug    bool
	Verbose  bool
	Port     string
	BindAddr string
	
	// Storage configuration
	StorageBackend string // "lmdb", "chdb", or "hybrid"
	LMDBPath       string
	ChDBPath       string
	
	// Hybrid storage configuration
	HybridConfig HybridConfig
	
	// chDB specific configuration
	ChDBConfig ChDBConfig
}

// HybridConfig contains configuration for hybrid storage routing
type HybridConfig struct {
	// Thresholds for storage backend selection
	HotDataThreshold    int64 // Max rows for LMDB (default: 1M)
	AnalyticalThreshold int64 // Min rows for chDB (default: 10M)
	
	// Auto-migration settings
	EnableAutoMigration bool  // Automatically migrate tables between backends
	MigrationCheckHours int   // Hours between migration checks
}

// ChDBConfig contains chDB-specific configuration
type ChDBConfig struct {
	// Memory limits
	MaxMemory         string // e.g., "8G", "16G"
	MaxMemoryForUser  string // Per-user memory limit
	
	// Performance settings
	MaxThreads        int    // Max threads for query execution
	MaxPartitionsPerInsert int // For batch inserts
	
	// Storage settings
	EnableCompression bool   // Enable data compression
	CompressionMethod string // "lz4", "zstd", etc.
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		Debug:          false,
		Verbose:        false,
		Port:           "3306",
		BindAddr:       "127.0.0.1",
		StorageBackend: "hybrid",
		LMDBPath:       "./data",
		ChDBPath:       "./chdb_data",
		HybridConfig: HybridConfig{
			HotDataThreshold:    1_000_000,  // 1M rows
			AnalyticalThreshold: 10_000_000, // 10M rows
			EnableAutoMigration: false,
			MigrationCheckHours: 24,
		},
		ChDBConfig: ChDBConfig{
			MaxMemory:         "4G",
			MaxMemoryForUser:  "2G",
			MaxThreads:        4,
			MaxPartitionsPerInsert: 100,
			EnableCompression: true,
			CompressionMethod: "lz4",
		},
	}
}

// LoadFromFlags loads configuration from command-line flags
func LoadFromFlags() *Config {
	cfg := DefaultConfig()
	
	// Define flags
	flag.BoolVar(&cfg.Debug, "debug", cfg.Debug, "Enable debug mode")
	flag.BoolVar(&cfg.Verbose, "verbose", cfg.Verbose, "Enable verbose logging")
	flag.StringVar(&cfg.Port, "port", cfg.Port, "Server port")
	flag.StringVar(&cfg.BindAddr, "bind", cfg.BindAddr, "Bind address")
	flag.StringVar(&cfg.StorageBackend, "storage", cfg.StorageBackend, "Storage backend (lmdb, chdb, hybrid)")
	flag.StringVar(&cfg.LMDBPath, "lmdb-path", cfg.LMDBPath, "LMDB data directory")
	flag.StringVar(&cfg.ChDBPath, "chdb-path", cfg.ChDBPath, "chDB data directory")
	
	// Hybrid storage flags
	flag.Int64Var(&cfg.HybridConfig.HotDataThreshold, "hot-data-threshold", cfg.HybridConfig.HotDataThreshold, "Max rows for LMDB storage")
	flag.Int64Var(&cfg.HybridConfig.AnalyticalThreshold, "analytical-threshold", cfg.HybridConfig.AnalyticalThreshold, "Min rows for chDB storage")
	flag.BoolVar(&cfg.HybridConfig.EnableAutoMigration, "auto-migrate", cfg.HybridConfig.EnableAutoMigration, "Enable automatic table migration")
	
	// chDB flags
	flag.StringVar(&cfg.ChDBConfig.MaxMemory, "chdb-max-memory", cfg.ChDBConfig.MaxMemory, "chDB max memory")
	flag.IntVar(&cfg.ChDBConfig.MaxThreads, "chdb-max-threads", cfg.ChDBConfig.MaxThreads, "chDB max threads")
	flag.BoolVar(&cfg.ChDBConfig.EnableCompression, "chdb-compression", cfg.ChDBConfig.EnableCompression, "Enable chDB compression")
	
	flag.Parse()
	
	// Override with environment variables if set
	cfg.LoadFromEnv()
	
	return cfg
}

// LoadFromEnv loads configuration from environment variables
func (c *Config) LoadFromEnv() {
	// Server settings
	if val := os.Getenv("DEBUG"); val != "" {
		c.Debug = val == "true" || val == "1"
	}
	if val := os.Getenv("VERBOSE"); val != "" {
		c.Verbose = val == "true" || val == "1"
	}
	if val := os.Getenv("PORT"); val != "" {
		c.Port = val
	}
	if val := os.Getenv("BIND_ADDR"); val != "" {
		c.BindAddr = val
	}
	
	// Storage settings
	if val := os.Getenv("STORAGE_BACKEND"); val != "" {
		c.StorageBackend = val
	}
	if val := os.Getenv("LMDB_PATH"); val != "" {
		c.LMDBPath = val
	}
	if val := os.Getenv("CHDB_PATH"); val != "" {
		c.ChDBPath = val
	}
	
	// Hybrid settings
	if val := os.Getenv("HOT_DATA_THRESHOLD"); val != "" {
		if threshold, err := strconv.ParseInt(val, 10, 64); err == nil {
			c.HybridConfig.HotDataThreshold = threshold
		}
	}
	if val := os.Getenv("ANALYTICAL_THRESHOLD"); val != "" {
		if threshold, err := strconv.ParseInt(val, 10, 64); err == nil {
			c.HybridConfig.AnalyticalThreshold = threshold
		}
	}
	
	// chDB settings
	if val := os.Getenv("CHDB_MAX_MEMORY"); val != "" {
		c.ChDBConfig.MaxMemory = val
	}
	if val := os.Getenv("CHDB_MAX_THREADS"); val != "" {
		if threads, err := strconv.Atoi(val); err == nil {
			c.ChDBConfig.MaxThreads = threads
		}
	}
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	// Validate storage backend
	switch c.StorageBackend {
	case "lmdb", "chdb", "hybrid":
		// Valid backends
	default:
		return flag.ErrHelp
	}
	
	// Validate thresholds
	if c.HybridConfig.HotDataThreshold >= c.HybridConfig.AnalyticalThreshold {
		c.HybridConfig.AnalyticalThreshold = c.HybridConfig.HotDataThreshold * 10
	}
	
	return nil
}