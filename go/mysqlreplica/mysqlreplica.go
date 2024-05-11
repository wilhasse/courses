package main

import (
	"context"
	"fmt"
	"os"

	"github.com/go-mysql-org/go-mysql/mysql"
	"github.com/go-mysql-org/go-mysql/replication"
)

func main() {
	cfg := replication.BinlogSyncerConfig{
		ServerID: 100,
		Flavor:   "mysql",
		Host:     "10.0.10.7",
		Port:     33307,
		User:     "root",
		Password: "07farm",
	}

	syncer := replication.NewBinlogSyncer(cfg)
	gtidSetString := "cf4426b2-f414-11ee-91aa-525400a7b1f7:1-125123341"

	// Parse the GTID set
	gtidSet, err := mysql.ParseMysqlGTIDSet(gtidSetString)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse GTID set: %v\n", err)
		return
	}

	// Start sync with specified GTID set
	streamer, err := syncer.StartSyncGTID(gtidSet)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start sync: %v\n", err)
		return
	}

	for {
		ev, err := streamer.GetEvent(context.Background())
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to get event: %v\n", err)
			continue
		}

		// Dump event
		//ev.Dump(os.Stdout)

		// Check the type of the event and handle row events
		switch e := ev.Event.(type) {
		case *replication.RowsEvent:
			handleRowEvent(ev.Header.EventType, e)
		}
	}
}

func handleRowEvent(eventType replication.EventType, event *replication.RowsEvent) {
	fmt.Printf("Schema: %s, Table: %s\n", event.Table.Schema, event.Table.Table)

	// Determine the type of row event based on the header event type
	switch eventType {
	case replication.WRITE_ROWS_EVENTv0, replication.WRITE_ROWS_EVENTv1, replication.WRITE_ROWS_EVENTv2:
		fmt.Println("Operation: INSERT")
		for _, row := range event.Rows {
			fmt.Println("Row data (new):", row)
		}

	case replication.UPDATE_ROWS_EVENTv0, replication.UPDATE_ROWS_EVENTv1, replication.UPDATE_ROWS_EVENTv2:
		fmt.Println("Operation: UPDATE")
		for i := 0; i < len(event.Rows); i += 2 {
			fmt.Println("Row data (old):", event.Rows[i])
			fmt.Println("Row data (new):", event.Rows[i+1])
		}

	case replication.DELETE_ROWS_EVENTv0, replication.DELETE_ROWS_EVENTv1, replication.DELETE_ROWS_EVENTv2:
		fmt.Println("Operation: DELETE")
		for _, row := range event.Rows {
			fmt.Println("Row data (old):", row)
		}
	}
}
