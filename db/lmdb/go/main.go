package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/rs/zerolog"
	"wellquite.org/golmdb"
)

func main() {
	// Create a directory for the database
	dbPath := "./testdb"
	err := os.MkdirAll(dbPath, 0755)
	if err != nil {
		log.Fatal("Failed to create database directory:", err)
	}

	// Create a logger
	logger := zerolog.New(os.Stdout).With().Timestamp().Logger()

	// Create LMDB client
	client, err := golmdb.NewLMDB(
		logger,
		dbPath,
		0644,
		126,    // max readers
		10,     // max DBs
		0,      // environment flags
		100,    // batch size
	)
	if err != nil {
		log.Fatal("Failed to create LMDB client:", err)
	}
	defer client.TerminateSync()

	// Generate random data
	rand.Seed(time.Now().UnixNano())
	timestamp := time.Now().Format("15:04:05")
	
	// Write some data using a write transaction
	fmt.Println("Writing random data to LMDB...")
	
	// Store the data we're writing so we can display it
	writtenData := make(map[string]string)
	
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		// Open or create a database
		db, err := txn.DBRef("mydb", golmdb.Create)
		if err != nil {
			return err
		}

		// Write 3 entries with random values
		for i := 1; i <= 3; i++ {
			key := fmt.Sprintf("key%d", i)
			value := fmt.Sprintf("random_%d_at_%s", rand.Intn(1000), timestamp)
			writtenData[key] = value
			
			fmt.Printf("  Writing: %s = %s\n", key, value)
			err = txn.Put(db, []byte(key), []byte(value), 0)
			if err != nil {
				return err
			}
		}

		// Add a new entry with incremental key
		newKey := fmt.Sprintf("entry_%d", time.Now().Unix())
		newValue := fmt.Sprintf("value_%d", rand.Intn(10000))
		writtenData[newKey] = newValue
		
		fmt.Printf("  Writing: %s = %s\n", newKey, newValue)
		err = txn.Put(db, []byte(newKey), []byte(newValue), 0)
		if err != nil {
			return err
		}

		return nil
	})
	if err != nil {
		log.Fatal("Failed to write data:", err)
	}

	// Read the data back using a read transaction
	fmt.Println("\nReading data from LMDB...")
	err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
		// Open the database for reading
		db, err := txn.DBRef("mydb", 0)
		if err != nil {
			return err
		}

		val, err := txn.Get(db, []byte("key1"))
		if err != nil {
			return err
		}
		fmt.Printf("key1: %s\n", string(val))

		val, err = txn.Get(db, []byte("key2"))
		if err != nil {
			return err
		}
		fmt.Printf("key2: %s\n", string(val))

		val, err = txn.Get(db, []byte("key3"))
		if err != nil {
			return err
		}
		fmt.Printf("key3: %s\n", string(val))

		// Use a cursor to iterate through all key-value pairs
		fmt.Println("\nIterating through ALL database entries using cursor:")
		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		// Position at first key
		totalEntries := 0
		key, val, err := cursor.First()
		if err == nil {
			fmt.Printf("  %s: %s\n", string(key), string(val))
			totalEntries++
			
			// Iterate through remaining keys
			for {
				key, val, err = cursor.Next()
				if err != nil {
					break
				}
				fmt.Printf("  %s: %s\n", string(key), string(val))
				totalEntries++
			}
		}
		fmt.Printf("\nTotal entries in database: %d\n", totalEntries)

		return nil
	})
	if err != nil {
		log.Fatal("Failed to read data:", err)
	}

	fmt.Println("\nExample completed successfully!")
}