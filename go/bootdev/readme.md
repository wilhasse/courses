# Course

Boot.Dev course with some exercises

This directory is not a entire project: go build ./... doesn't work
It is a collection of exercises that I did from the Boot.Dev course


## Hello World

Hello World

```bash
go build -o bootdev ./cmd
```

## Variables

```bash
go build -o test_basic ./variables/basic.go
./test_basic
go build -o test_balance ./variables/balance.go
./test_balance
```

## Install Bootdev Tools

```bash
# Install
go install github.com/bootdotdev/bootdev@latest

# See bin path
go env GOPATH

# /home/cslog/go
# binary would be in /home/cslog/go/bin
bootdev
```

## Other Units (test individually)

Test one file inside directory

```bash
go test -v ./interfaces/formatter.go ./interfaces/formatter_test.go
```

## Example (project)

Test one directory (all files)

```bash
go test -v ./example/
```

Test all files

```bash
go test -v ./...
```

