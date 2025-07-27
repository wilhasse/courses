package benchmark

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/rs/zerolog"
	"wellquite.org/golmdb"
)

const (
	dbPath   = "./benchmark_testdb"
	dbName   = "benchdb"
	capacity = 128 // Equivalent to B+ tree capacity
)

// BenchmarkResult stores benchmark timing information
type BenchmarkResult struct {
	Name        string
	Operations  int
	TimeMicros  float64
	OpsPerSec   float64
	MicrosPerOp float64
}

// Helper function to create LMDB client for benchmarks
func createClient() (*golmdb.LMDBClient, error) {
	// Clean up any existing database
	os.RemoveAll(dbPath)
	os.MkdirAll(dbPath, 0755)

	logger := zerolog.New(os.Stdout).Level(zerolog.ErrorLevel) // Quiet logging for benchmarks
	
	client, err := golmdb.NewLMDB(
		logger,
		dbPath,
		0644,
		126,    // max readers
		10,     // max DBs
		0,      // environment flags
		1000,   // batch size for better performance
	)
	return client, err
}

// Helper to convert int to bytes (little endian)
func intToBytes(i int) []byte {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, uint64(i))
	return b
}

// Helper to convert bytes back to int
func bytesToInt(b []byte) int {
	return int(binary.LittleEndian.Uint64(b))
}

// Sequential insertion benchmark - equivalent to BPlusTree sequential insert
func BenchmarkSequentialInsert(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				client, err := createClient()
				if err != nil {
					b.Fatal("Failed to create LMDB client:", err)
				}
				b.StartTimer()
				
				err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
					db, err := txn.DBRef(dbName, golmdb.Create)
					if err != nil {
						return err
					}
					
					for j := 0; j < size; j++ {
						key := intToBytes(j)
						value := intToBytes(j * 2)
						err = txn.Put(db, key, value, 0)
						if err != nil {
							return err
						}
					}
					return nil
				})
				
				b.StopTimer()
				if err != nil {
					b.Fatal("Failed to insert data:", err)
				}
				client.TerminateSync()
				os.RemoveAll(dbPath)
				b.StartTimer()
			}
		})
	}
}

// Random insertion benchmark - equivalent to BPlusTree random insert
func BenchmarkRandomInsert(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			// Pre-generate random keys
			rng := rand.New(rand.NewSource(12345))
			keys := make([]int, size)
			for i := range keys {
				keys[i] = rng.Int()
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				client, err := createClient()
				if err != nil {
					b.Fatal("Failed to create LMDB client:", err)
				}
				b.StartTimer()
				
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
				
				b.StopTimer()
				if err != nil {
					b.Fatal("Failed to insert data:", err)
				}
				client.TerminateSync()
				os.RemoveAll(dbPath)
				b.StartTimer()
			}
		})
	}
}

// Lookup benchmark - equivalent to BPlusTree random lookup
func BenchmarkLookup(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			// Setup: create client and populate data
			client, err := createClient()
			if err != nil {
				b.Fatal("Failed to create LMDB client:", err)
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
				
				for j := 0; j < size; j++ {
					key := intToBytes(j)
					value := intToBytes(j * 2)
					err = txn.Put(db, key, value, 0)
					if err != nil {
						return err
					}
				}
				return nil
			})
			if err != nil {
				b.Fatal("Failed to populate data:", err)
			}
			
			// Pre-generate random lookup keys
			rng := rand.New(rand.NewSource(54321))
			lookupCount := min(1000, size)
			lookupKeys := make([][]byte, lookupCount)
			for i := range lookupKeys {
				lookupKeys[i] = intToBytes(rng.Intn(size))
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
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
				if err != nil {
					b.Fatal("Failed to lookup data:", err)
				}
			}
		})
	}
}

// Iteration benchmark - equivalent to BPlusTree full iteration
func BenchmarkIteration(b *testing.B) {
	sizes := []int{100, 1000, 10000} // Limit to smaller sizes like B+ tree benchmarks
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			// Setup: create client and populate data
			client, err := createClient()
			if err != nil {
				b.Fatal("Failed to create LMDB client:", err)
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
				
				for j := 0; j < size; j++ {
					key := intToBytes(j)
					value := intToBytes(j * 2)
					err = txn.Put(db, key, value, 0)
					if err != nil {
						return err
					}
				}
				return nil
			})
			if err != nil {
				b.Fatal("Failed to populate data:", err)
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
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
					
					if count != size {
						return fmt.Errorf("expected %d items, got %d", size, count)
					}
					return nil
				})
				if err != nil {
					b.Fatal("Failed to iterate data:", err)
				}
			}
		})
	}
}

// Range query benchmark - simulated with cursor range iteration
func BenchmarkRangeQuery(b *testing.B) {
	sizes := []int{100, 1000, 10000} // Limit to smaller sizes
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			// Setup: create client and populate data
			client, err := createClient()
			if err != nil {
				b.Fatal("Failed to create LMDB client:", err)
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
				
				for j := 0; j < size; j++ {
					key := intToBytes(j)
					value := intToBytes(j * 2)
					err = txn.Put(db, key, value, 0)
					if err != nil {
						return err
					}
				}
				return nil
			})
			if err != nil {
				b.Fatal("Failed to populate data:", err)
			}
			
			// Define range: start at size/4, get size/10 elements (like B+ tree benchmark)
			start := size / 4
			rangeSize := size / 10
			if rangeSize < 1 {
				rangeSize = 1
			}
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
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
					
					// Position cursor at start key
					startKey := intToBytes(start)
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
					b.Fatal("Failed to range query:", err)
				}
			}
		})
	}
}

// Comparison benchmark - LMDB vs Go map (like the B+ tree comparison)
func BenchmarkComparison(b *testing.B) {
	sizes := []int{100, 1000, 10000, 100000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("Size-%d", size), func(b *testing.B) {
			// Prepare data
			keys := make([]int, size)
			for i := range keys {
				keys[i] = i
			}
			
			// Shuffle for random access
			shuffled := make([]int, size)
			copy(shuffled, keys)
			rand.Shuffle(len(shuffled), func(i, j int) {
				shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
			})
			
			// Sequential Insertion - LMDB
			b.Run("SequentialInsert/LMDB", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					client, err := createClient()
					if err != nil {
						b.Fatal("Failed to create LMDB client:", err)
					}
					b.StartTimer()
					
					err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
						db, err := txn.DBRef(dbName, golmdb.Create)
						if err != nil {
							return err
						}
						
						for j := 0; j < size; j++ {
							key := intToBytes(j)
							value := intToBytes(j * 2)
							err = txn.Put(db, key, value, 0)
							if err != nil {
								return err
							}
						}
						return nil
					})
					
					b.StopTimer()
					if err != nil {
						b.Fatal("Failed to insert data:", err)
					}
					client.TerminateSync()
					os.RemoveAll(dbPath)
					b.StartTimer()
				}
			})
			
			// Sequential Insertion - Go Map
			b.Run("SequentialInsert/Map", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					m := make(map[int]int, size)
					for j := 0; j < size; j++ {
						m[j] = j * 2
					}
				}
			})
			
			// Random Insertion - LMDB
			b.Run("RandomInsert/LMDB", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					client, err := createClient()
					if err != nil {
						b.Fatal("Failed to create LMDB client:", err)
					}
					b.StartTimer()
					
					err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
						db, err := txn.DBRef(dbName, golmdb.Create)
						if err != nil {
							return err
						}
						
						for _, k := range shuffled {
							key := intToBytes(k)
							value := intToBytes(k * 2)
							err = txn.Put(db, key, value, 0)
							if err != nil {
								return err
							}
						}
						return nil
					})
					
					b.StopTimer()
					if err != nil {
						b.Fatal("Failed to insert data:", err)
					}
					client.TerminateSync()
					os.RemoveAll(dbPath)
					b.StartTimer()
				}
			})
			
			// Random Insertion - Go Map
			b.Run("RandomInsert/Map", func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					m := make(map[int]int, size)
					for _, k := range shuffled {
						m[k] = k * 2
					}
				}
			})
			
			// Setup populated structures for lookup tests
			client, err := createClient()
			if err != nil {
				b.Fatal("Failed to create LMDB client:", err)
			}
			defer func() {
				client.TerminateSync()
				os.RemoveAll(dbPath)
			}()
			
			err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
				db, err := txn.DBRef(dbName, golmdb.Create)
				if err != nil {
					return err
				}
				
				for i := 0; i < size; i++ {
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
				b.Fatal("Failed to populate LMDB:", err)
			}
			
			m := make(map[int]int, size)
			for i := 0; i < size; i++ {
				m[i] = i * 2
			}
			
			// Lookup - LMDB
			b.Run("Lookup/LMDB", func(b *testing.B) {
				lookupKeys := shuffled[:min(1000, size)]
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
						db, err := txn.DBRef(dbName, 0)
						if err != nil {
							return err
						}
						
						for _, k := range lookupKeys {
							key := intToBytes(k)
							_, err = txn.Get(db, key)
							if err != nil {
								return err
							}
						}
						return nil
					})
					if err != nil {
						b.Fatal("Failed to lookup:", err)
					}
				}
			})
			
			// Lookup - Go Map
			b.Run("Lookup/Map", func(b *testing.B) {
				lookupKeys := shuffled[:min(1000, size)]
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for _, k := range lookupKeys {
						_ = m[k]
					}
				}
			})
			
			// Iteration tests (limited to smaller sizes)
			if size <= 10000 {
				// Iteration - LMDB
				b.Run("Iteration/LMDB", func(b *testing.B) {
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
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
							
							if count != size {
								return fmt.Errorf("expected %d items, got %d", size, count)
							}
							return nil
						})
						if err != nil {
							b.Fatal("Failed to iterate:", err)
						}
					}
				})
				
				// Iteration - Go Map
				b.Run("Iteration/Map", func(b *testing.B) {
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						count := 0
						for range m {
							count++
						}
						if count != size {
							b.Fatalf("Expected %d items, got %d", size, count)
						}
					}
				})
				
				// Range Query - LMDB only (Go maps don't support efficient range queries)
				b.Run("RangeQuery/LMDB", func(b *testing.B) {
					start := size / 4
					rangeSize := size / 10
					if rangeSize < 1 {
						rangeSize = 1
					}
					
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
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
							
							// Position cursor at start key
							startKey := intToBytes(start)
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
							b.Fatal("Failed to range query:", err)
						}
					}
				})
			}
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}