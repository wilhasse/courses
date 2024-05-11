# Projects

| Name            | Description                            |
| ----------------|----------------------------------------|
| database        | Book: Build your own                   |
| hello           | Hello World                            |
| http            | Basic Http Server                      |  
| memsql          | Memory SQL Server (go-mysql-server)    |
| mysqlreplica    | Replication test from mysql            |
| testsql         | Parsing SQL using Vitess               |

# Basic Commands

Create new module

```bash
mkdir project
cd project
go mod init project
```

Correct Syntax  all files

```bash
go fmt ./...
```

Organize packages

```bash
go mod tidy
```
Compile and create executable

```bash
go build
```

Run only

```bash
go run project
```
