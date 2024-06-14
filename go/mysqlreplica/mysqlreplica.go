package main

import (
    "context"
    "database/sql"
    "fmt"
    "os"
    "time"
    "strings"

    "github.com/go-mysql-org/go-mysql/mysql"
    "github.com/go-mysql-org/go-mysql/replication"
    _ "github.com/go-sql-driver/mysql"
)

// thiread_id
var thread_id uint32 = 0

func main() {
    // Database connection string
    db, err := sql.Open("mysql", "root:07farm@tcp(10.1.0.10:3306)/teste_repl")
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to connect to database: %v\n", err)
        return
    }
    defer db.Close()

    cfg := replication.BinlogSyncerConfig{
        ServerID: 100,
        Flavor:   "mysql",
        Host:     "10.1.0.7",
        Port:     3306,
        User:     "root",
        Password: "07farm",
    }

    syncer := replication.NewBinlogSyncer(cfg)
    gtidSetString := os.Args[1]

    gtidSet, err := mysql.ParseMysqlGTIDSet(gtidSetString)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to parse GTID set: %v\n", err)
        return
    }

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

        switch e := ev.Event.(type) {
        case *replication.RowsEvent:
            handleRowEvent(db, ev.Header.EventType, e,  ev.Header.Timestamp, syncer.GetNextPosition().Name, ev.Header.LogPos)
        case *replication.QueryEvent:
	    //ev.Dump(os.Stdout)
            handleQueryEvent(db, e, ev.Header.LogPos)
        }
    }
}

func handleRowEvent(db *sql.DB, eventType replication.EventType, event *replication.RowsEvent, eventDate uint32, logName string, logPos uint32) {
    var operationType int
    var oldData, newData string

    switch eventType {
    case replication.WRITE_ROWS_EVENTv0, replication.WRITE_ROWS_EVENTv1, replication.WRITE_ROWS_EVENTv2:
        operationType = 1
        newData = fmt.Sprintf("%v", event.Rows)

    case replication.UPDATE_ROWS_EVENTv0, replication.UPDATE_ROWS_EVENTv1, replication.UPDATE_ROWS_EVENTv2:
        operationType = 2
        for i := 0; i < len(event.Rows); i += 2 {
            oldData = fmt.Sprintf("%v", event.Rows[i])
            newData = fmt.Sprintf("%v", event.Rows[i+1])
        }

    case replication.DELETE_ROWS_EVENTv0, replication.DELETE_ROWS_EVENTv1, replication.DELETE_ROWS_EVENTv2:
        operationType = 3
        oldData = fmt.Sprintf("%v", event.Rows)
    }

    tableName := event.Table.Table
    eventTime := time.Unix(int64(eventDate), 0)

    // Subtract 3 hours from the event time
    adjustedTime := eventTime.Add(-3 * time.Hour)

    if string(tableName) != "LAYOUT" {
        _, err := db.Exec("INSERT INTO BINLOG_EVENTS (DATA, DML, TABLENAME, NEW_DATA, OLD_DATA, THREAD_ID, BINLOG_NAME, BINLOG_POS) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  adjustedTime, operationType, tableName, newData, oldData, thread_id, logName, logPos)

        if err != nil {
            fmt.Fprintf(os.Stderr, "Failed to insert event data: %v\n", err)
        }
    }
}

func handleQueryEvent(db *sql.DB, event *replication.QueryEvent, logPos uint32) {
    query := string(event.Query)
    eventTime := time.Unix(int64(event.ExecutionTime), 0)  // Convert UNIX timestamp to time.Time

    // Subtract 3 hours from the event time
    adjustedTime := eventTime.Add(-3 * time.Hour)

    // Convert query to upper case to ensure case-insensitive comparison
    upperQuery := strings.ToUpper(query)

    // Update ThreadId
    if strings.Contains(upperQuery, "BEGIN") {
       thread_id = event.SlaveProxyID
    }

    // Check if the query is a "BEGIN", "COMMIT", or other non-critical event
    if strings.Contains(upperQuery, "BEGIN") || strings.Contains(upperQuery, "COMMIT") || strings.Contains(upperQuery, "SET") {
        // Ignore these events
        return
    }

    // Log all other query events
    _, err := db.Exec("INSERT INTO BINLOG_EVENTS (DATA, DML, TABLENAME, NEW_DATA) VALUES (?, ?, ?, ?)",
        adjustedTime, 4, "", query)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Failed to insert query event data: %v\n", err)
    }
}
