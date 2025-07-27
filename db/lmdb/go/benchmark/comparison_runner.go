package benchmark

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/rs/zerolog"
	"wellquite.org/golmdb"
)

const (
	dbPath   = "./benchmark_testdb"
	dbName   = "benchdb"
)

type BenchmarkResult struct {
	Name        string
	Operations  int
	TimeMicros  float64
	OpsPerSec   float64
	MicrosPerOp float64
}

func printResult(result BenchmarkResult) {
	fmt.Printf("%-30s %10d ops in %8.2fms | %12.0f ops/sec | %8.2f µs/op\n",
		result.Name,
		result.Operations,
		result.TimeMicros/1000, // Convert µs to ms for display
		result.OpsPerSec,
		result.MicrosPerOp,
	)
}

func createClient() (*golmdb.LMDBClient, error) {
	// Clean up any existing database
	os.RemoveAll(dbPath)
	os.MkdirAll(dbPath, 0755)

	logger := zerolog.New(os.Stdout).Level(zerolog.ErrorLevel) // Quiet logging

	client, err := golmdb.NewLMDB(
		logger,
		dbPath,
		0644,
		126,  // max readers
		10,   // max DBs
		0,    // environment flags
		1000, // batch size
	)
	return client, err
}

func intToBytes(i int) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, uint64(i))
	return b
}

func benchmarkSequentialInsert(n int) BenchmarkResult {
	client, err := createClient()
	if err != nil {
		panic("Failed to create LMDB client: " + err.Error())
	}
	defer func() {
		client.TerminateSync()
		os.RemoveAll(dbPath)
	}()

	start := time.Now()
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef(dbName, golmdb.Create)
		if err != nil {
			return err
		}

		for i := 0; i < n; i++ {
			key := intToBytes(i)
			value := intToBytes(i * 2)
			err = txn.Put(db, key, value, 0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	elapsed := time.Since(start)

	if err != nil {
		panic("Failed to insert data: " + err.Error())
	}

	timeMicros := float64(elapsed.Nanoseconds()) / 1000
	opsPerSec := float64(n) * 1_000_000 / timeMicros
	microsPerOp := timeMicros / float64(n)

	return BenchmarkResult{
		Name:        "Sequential insertion",
		Operations:  n,
		TimeMicros:  timeMicros,
		OpsPerSec:   opsPerSec,
		MicrosPerOp: microsPerOp,
	}
}

func benchmarkRandomInsert(n int) BenchmarkResult {
	client, err := createClient()
	if err != nil {
		panic("Failed to create LMDB client: " + err.Error())
	}
	defer func() {
		client.TerminateSync()
		os.RemoveAll(dbPath)
	}()

	// Pre-generate random keys
	rng := rand.New(rand.NewSource(12345))
	keys := make([]int, n)
	for i := range keys {
		keys[i] = rng.Int()
	}

	start := time.Now()
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef(dbName, golmdb.Create)
		if err != nil {
			return err
		}

		for _, key := range keys {
			keyBytes := intToBytes(key)
			valueBytes := intToBytes(key * 2)
			err = txn.Put(db, keyBytes, valueBytes, 0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	elapsed := time.Since(start)

	if err != nil {
		panic("Failed to insert data: " + err.Error())
	}

	timeMicros := float64(elapsed.Nanoseconds()) / 1000
	opsPerSec := float64(n) * 1_000_000 / timeMicros
	microsPerOp := timeMicros / float64(n)

	return BenchmarkResult{
		Name:        "Random insertion",
		Operations:  n,
		TimeMicros:  timeMicros,
		OpsPerSec:   opsPerSec,
		MicrosPerOp: microsPerOp,
	}
}

func benchmarkLookup(n int) BenchmarkResult {
	client, err := createClient()
	if err != nil {
		panic("Failed to create LMDB client: " + err.Error())
	}
	defer func() {
		client.TerminateSync()
		os.RemoveAll(dbPath)
	}()

	// Populate with sequential data
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef(dbName, golmdb.Create)
		if err != nil {
			return err
		}

		for i := 0; i < n; i++ {
			key := intToBytes(i)
			value := intToBytes(i * 2)
			err = txn.Put(db, key, value, 0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		panic("Failed to populate data: " + err.Error())
	}

	// Pre-generate random lookup keys
	rng := rand.New(rand.NewSource(54321))
	lookupKeys := make([][]byte, n)
	for i := range lookupKeys {
		lookupKeys[i] = intToBytes(rng.Intn(n))
	}

	start := time.Now()
	err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef(dbName, 0)
		if err != nil {
			return err
		}

		for _, key := range lookupKeys {
			_, err = txn.Get(db, key)
			if err != nil {
				return err
			}
		}
		return nil
	})
	elapsed := time.Since(start)

	if err != nil {
		panic("Failed to lookup data: " + err.Error())
	}

	timeMicros := float64(elapsed.Nanoseconds()) / 1000
	opsPerSec := float64(n) * 1_000_000 / timeMicros
	microsPerOp := timeMicros / float64(n)

	return BenchmarkResult{
		Name:        "Random lookup (hit)",
		Operations:  n,
		TimeMicros:  timeMicros,
		OpsPerSec:   opsPerSec,
		MicrosPerOp: microsPerOp,
	}
}

func benchmarkIteration(n int) BenchmarkResult {
	client, err := createClient()
	if err != nil {
		panic("Failed to create LMDB client: " + err.Error())
	}
	defer func() {
		client.TerminateSync()
		os.RemoveAll(dbPath)
	}()

	// Populate with sequential data
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef(dbName, golmdb.Create)
		if err != nil {
			return err
		}

		for i := 0; i < n; i++ {
			key := intToBytes(i)
			value := intToBytes(i * 2)
			err = txn.Put(db, key, value, 0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		panic("Failed to populate data: " + err.Error())
	}

	start := time.Now()
	err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef(dbName, 0)
		if err != nil {
			return err
		}

		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		count := 0
		_, _, err = cursor.First()
		if err == nil {
			count++
			for {
				_, _, err = cursor.Next()
				if err != nil {
					break
				}
				count++
			}
		}

		if count != n {
			return fmt.Errorf("expected %d items, got %d", n, count)
		}
		return nil
	})
	elapsed := time.Since(start)

	if err != nil {
		panic("Failed to iterate data: " + err.Error())
	}

	timeMicros := float64(elapsed.Nanoseconds()) / 1000
	opsPerSec := float64(n) * 1_000_000 / timeMicros
	microsPerOp := timeMicros / float64(n)

	return BenchmarkResult{
		Name:        "Full iteration",
		Operations:  n,
		TimeMicros:  timeMicros,
		OpsPerSec:   opsPerSec,
		MicrosPerOp: microsPerOp,
	}
}

func benchmarkRangeQuery(n int) BenchmarkResult {
	client, err := createClient()
	if err != nil {
		panic("Failed to create LMDB client: " + err.Error())
	}
	defer func() {
		client.TerminateSync()
		os.RemoveAll(dbPath)
	}()

	// Populate with sequential data
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		db, err := txn.DBRef(dbName, golmdb.Create)
		if err != nil {
			return err
		}

		for i := 0; i < n; i++ {
			key := intToBytes(i)
			value := intToBytes(i * 2)
			err = txn.Put(db, key, value, 0)
			if err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		panic("Failed to populate data: " + err.Error())
	}

	// Range parameters: like B+ tree benchmark (1000 items, 100 queries)
	rangeSize := 1000
	numRanges := 100

	start := time.Now()
	for i := 0; i < numRanges; i++ {
		err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
			db, err := txn.DBRef(dbName, 0)
			if err != nil {
				return err
			}

			cursor, err := txn.NewCursor(db)
			if err != nil {
				return err
			}
			defer cursor.Close()

			// Position cursor at start of range
			startIdx := i * (n / numRanges)
			startKey := intToBytes(startIdx)
			_, _, err = cursor.SeekGreaterThanOrEqualKey(startKey)
			if err != nil {
				// If no key found, position at first key
				_, _, err = cursor.First()
				if err != nil {
					return err
				}
			}

			count := 0
			for count < rangeSize {
				_, _, err = cursor.Current()
				if err != nil {
					break
				}
				count++
				_, _, err = cursor.Next()
				if err != nil {
					break
				}
			}

			return nil
		})
		if err != nil {
			panic("Failed to range query: " + err.Error())
		}
	}
	elapsed := time.Since(start)

	timeMicros := float64(elapsed.Nanoseconds()) / 1000
	opsPerSec := float64(numRanges) * 1_000_000 / timeMicros
	microsPerOp := timeMicros / float64(numRanges)

	return BenchmarkResult{
		Name:        "Range query (1000 items)",
		Operations:  numRanges,
		TimeMicros:  timeMicros,
		OpsPerSec:   opsPerSec,
		MicrosPerOp: microsPerOp,
	}
}

func main() {
	fmt.Println()
	fmt.Println("=== LMDB Performance Benchmarks ===")
	fmt.Println()

	// Test with the same sizes as B+ tree benchmarks
	sizes := []int{100, 1000, 10000, 100000}

	fmt.Println("Operation benchmarks (standardized for comparison with B+ Tree):")
	fmt.Println("----------------------------------------------------------------")

	for _, n := range sizes {
		fmt.Printf("\n--- Dataset size: %d ---\n", n)

		// Sequential insertion
		result := benchmarkSequentialInsert(n)
		printResult(result)

		// Random insertion
		result = benchmarkRandomInsert(n)
		printResult(result)

		// Random lookup
		result = benchmarkLookup(n)
		printResult(result)

		// Full iteration (limit to smaller sizes for performance)
		if n <= 10000 {
			result = benchmarkIteration(n)
			printResult(result)
		}

		// Range query (limit to smaller sizes)
		if n <= 10000 {
			result = benchmarkRangeQuery(n)
			printResult(result)
		}
	}

	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Println("LMDB Characteristics:")
	fmt.Println("  ✓ ACID transactions")
	fmt.Println("  ✓ Memory-mapped storage")
	fmt.Println("  ✓ Ordered key iteration (B+ tree structure)")
	fmt.Println("  ✓ Range queries via cursors")
	fmt.Println("  ✓ Persistent storage")
	fmt.Println("  ✓ Copy-on-write (zero-copy reads)")
	fmt.Println("  ✓ Multiple concurrent readers")
	fmt.Println()
	fmt.Println("Compare these results with B+ Tree benchmarks:")
	fmt.Println("  cd /home/cslog/BPlusTree3")
	fmt.Println("  ./scripts/run_all_benchmarks.py -v")
	fmt.Println()
}