package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"syscall"
	"time"

	"github.com/dolthub/dolt/go/libraries/doltcore/sqle/binlogreplication"
	"github.com/dolthub/go-mysql-server/sql"
	mysqlbinlog "github.com/dolthub/go-mysql-server/sql/binlogreplication"
	"github.com/jmoiron/sqlx"
        _ "github.com/go-sql-driver/mysql"
)

var mySqlPort, doltPort int
var primaryDatabase, replicaDatabase *sqlx.DB
var mySqlProcess, doltProcess *os.Process
var doltLogFilePath, oldDoltLogFilePath, mysqlLogFilePath string
var doltLogFile, mysqlLogFile *os.File
var testDir string
var originalWorkingDir string

func waitForSqlServerToStart(database *sqlx.DB) error {
	fmt.Printf("Waiting for server to start...\n")
	for counter := 0; counter < 20; counter++ {
		if database.Ping() == nil {
			return nil
		}
		fmt.Printf("not up yet; waiting...\n")
		time.Sleep(500 * time.Millisecond)
	}

	return database.Ping()
}

func initializeDevDoltBuild(dir string, goDirPath string) string {
	// If we're not in a CI environment, don't worry about building a dev build
	if os.Getenv("CI") != "true" {
		return ""
	}

	basedir := filepath.Dir(filepath.Dir(dir))
	fullpath := filepath.Join(basedir, fmt.Sprintf("devDolt-%d", os.Getpid()))

	_, err := os.Stat(fullpath)
	if err == nil {
		return fullpath
	}

	fmt.Printf("building dolt dev build at: %s \n", fullpath)
	cmd := exec.Command("go", "build", "-o", fullpath, "./cmd/dolt")
	cmd.Dir = goDirPath

	output, err := cmd.CombinedOutput()
	if err != nil {
		panic("unable to build dolt for binlog integration tests: " + err.Error() + "\nFull output: " + string(output) + "\n")
	}
	return fullpath
}

func findFreePort() int {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		panic(fmt.Sprintf("unable to find available TCP port: %v", err.Error()))
	}
	mySqlPort := listener.Addr().(*net.TCPAddr).Port
	err = listener.Close()
	if err != nil {
		panic(fmt.Sprintf("unable to find available TCP port: %v", err.Error()))
	}

	return mySqlPort
}

func startDoltSqlServer(dir string) (int, *os.Process, error) {
	dir = filepath.Join(dir, "dolt")
	err := os.MkdirAll(dir, 0777)
	if err != nil {
		return -1, nil, err
	}

	doltPort = findFreePort()
	fmt.Printf("Starting Dolt sql-server on port: %d, with data dir %s\n", doltPort, dir)

	// take the CWD and move up four directories to find the go directory
	if originalWorkingDir == "" {
		var err error
		originalWorkingDir, err = os.Getwd()
		if err != nil {
			panic(err)
		}
	}
	goDirPath := filepath.Join(originalWorkingDir, "..", "..", "..", "..")
	err = os.Chdir(goDirPath)
	if err != nil {
		panic(err)
	}

	socketPath := filepath.Join("/tmp", fmt.Sprintf("dolt.%v.sock", doltPort))

	// use an admin user NOT named "root" to test that we don't require the "root" account
	adminUser := "admin"

	args := []string{"go", "run", "./cmd/dolt",
		"sql-server",
		fmt.Sprintf("-u%s", adminUser),
		"--loglevel=TRACE",
		fmt.Sprintf("--data-dir=%s", dir),
		fmt.Sprintf("--port=%v", doltPort),
		fmt.Sprintf("--socket=%s", socketPath)}

	// If we're running in CI, use a precompiled dolt binary instead of go run
	devDoltPath := initializeDevDoltBuild(dir, goDirPath)
	if devDoltPath != "" {
		args[2] = devDoltPath
		args = args[2:]
	}
	cmd := exec.Command(args[0], args[1:]...)

	// Set a unique process group ID so that we can cleanly kill this process, as well as
	// any spawned child processes later. Mac/Unix can set the "Setpgid" field directly, but
	// on windows, this field isn't present, so we need to use reflection so that this code
	// can still compile for windows, even though we don't run it there.
	procAttr := &syscall.SysProcAttr{}
	ps := reflect.ValueOf(procAttr)
	s := ps.Elem()
	f := s.FieldByName("Setpgid")
	f.SetBool(true)
	cmd.SysProcAttr = procAttr

	// Some tests restart the Dolt sql-server, so if we have a current log file, save a reference
	// to it so we can print the results later if the test fails.
	if doltLogFilePath != "" {
		oldDoltLogFilePath = doltLogFilePath
	}

	doltLogFilePath = filepath.Join(dir, fmt.Sprintf("dolt-%d.out.log", time.Now().Unix()))
	doltLogFile, err = os.Create(doltLogFilePath)
	if err != nil {
		return -1, nil, err
	}
	fmt.Printf("dolt sql-server logs at: %s \n", doltLogFilePath)
	cmd.Stdout = doltLogFile
	cmd.Stderr = doltLogFile
	err = cmd.Start()
	if err != nil {
		return -1, nil, fmt.Errorf("unable to execute command %v: %v", cmd.String(), err.Error())
	}

	fmt.Printf("Dolt CMD: %s\n", cmd.String())

	dsn := fmt.Sprintf("%s@tcp(127.0.0.1:%v)/", adminUser, doltPort)
	replicaDatabase = sqlx.MustOpen("mysql", dsn)

	err = waitForSqlServerToStart(replicaDatabase)
	if err != nil {
		return -1, nil, err
	}

	fmt.Printf("Dolt server started on port %v \n", doltPort)

	return doltPort, cmd.Process, nil
}

func main() {

	// Start the Dolt SQL server
	doltDir := "/home/cslog/dolt_data"
	var err error
	_, doltProcess, err = startDoltSqlServer(doltDir)
	if err != nil {
		log.Fatalf("Failed to start Dolt SQL server: %v", err)
	}
	defer doltProcess.Kill()

	// Create a new SQL context
	ctx := sql.NewContext(context.Background())

	// Set up the replication source options
	sourceOptions := []mysqlbinlog.ReplicationOption{
		{
			Name:  "SOURCE_HOST",
			Value: mysqlbinlog.StringReplicationOptionValue{Value: "10.1.0.9"},
		},
		{
			Name:  "SOURCE_USER",
			Value: mysqlbinlog.StringReplicationOptionValue{Value: "root"},
		},
		{
			Name:  "SOURCE_PASSWORD",
			Value: mysqlbinlog.StringReplicationOptionValue{Value: "07farm"},
		},
		{
			Name:  "SOURCE_PORT",
			Value: mysqlbinlog.IntegerReplicationOptionValue{Value: 3306},
		},
	}

	// Set up the binlog replica controller
	controller := binlogreplication.DoltBinlogReplicaController

	// Set the execution context
	controller.SetExecutionContext(ctx)

	// Set the replication source options
	err = controller.SetReplicationSourceOptions(ctx, sourceOptions)
	if err != nil {
		log.Fatalf("Failed to set replication source options: %v", err)
	}

	// Start the replica
	err = controller.StartReplica(ctx)
	if err != nil {
		log.Fatalf("Failed to start replica: %v", err)
	}

	log.Println("Replication started successfully")

	// Wait for some time to allow replication to happen
	time.Sleep(30 * time.Second)

	// Stop the replica
	err = controller.StopReplica(ctx)
	if err != nil {
		log.Fatalf("Failed to stop replica: %v", err)
	}

	log.Println("Replication stopped successfully")
}
