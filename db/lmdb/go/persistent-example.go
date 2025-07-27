package main

import (
	"fmt"
	"log"
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

	// First, try to read existing data
	fmt.Println("Checking for existing data...")
	existingCount := 0
	err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
		// Open the database for reading
		db, err := txn.DBRef("mydb", 0)
		if err != nil {
			fmt.Println("No existing database found")
			return nil
		}

		// Count existing entries
		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		_, _, err = cursor.First()
		if err != nil {
			fmt.Println("Database is empty")
			return nil
		}

		existingCount = 1
		for {
			_, _, err = cursor.Next()
			if err != nil {
				break
			}
			existingCount++
		}
		fmt.Printf("Found %d existing entries\n", existingCount)
		return nil
	})

	// Add a new entry with timestamp
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	newKey := fmt.Sprintf("key_%d", existingCount+1)
	newValue := fmt.Sprintf("Added at %s", timestamp)

	fmt.Printf("\nAdding new entry: %s = %s\n", newKey, newValue)
	err = client.Update(func(txn *golmdb.ReadWriteTxn) error {
		// Open or create a database
		db, err := txn.DBRef("mydb", golmdb.Create)
		if err != nil {
			return err
		}

		return txn.Put(db, []byte(newKey), []byte(newValue), 0)
	})
	if err != nil {
		log.Fatal("Failed to write data:", err)
	}

	// Read all data to show persistence
	fmt.Println("\nAll database entries:")
	err = client.View(func(txn *golmdb.ReadOnlyTxn) error {
		db, err := txn.DBRef("mydb", 0)
		if err != nil {
			return err
		}

		cursor, err := txn.NewCursor(db)
		if err != nil {
			return err
		}
		defer cursor.Close()

		key, val, err := cursor.First()
		if err == nil {
			fmt.Printf("  %s: %s\n", string(key), string(val))
			
			for {
				key, val, err = cursor.Next()
				if err != nil {
					break
				}
				fmt.Printf("  %s: %s\n", string(key), string(val))
			}
		}
		return nil
	})
	if err != nil {
		log.Fatal("Failed to read data:", err)
	}

	fmt.Println("\nRun this program again to see data persistence!")
}