# Description

Calling a C function in Go

## Files

- add.h
- add.c
- go_c.go

## Commands

generate go.mod

```bash
//init go project
go mod init go_c
```

## Build

normal

```bash
CGO_ENABLED=1 GOOS=linux CC="zig cc -target x86_64-linux" go build .
```

uses musl libc

```bash
CGO_ENABLED=1 GOOS=linux CC="zig cc -target x86_64-linux-musl" go build .
```

compiling only add.c to generate obj
there is no main function: zig cc add.c will genereate an error

```bash
zig cc -c -target x86_64-linux add.c -o add.o
```
