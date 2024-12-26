# Source Code

## [mysql-server](./mysql-server)

Simple mysql server that respond to SELECT 1

## [xplugin-test](./xplugin-test)

Testing get a list of databases using native MySQL X Plugin

## [select-test](./select-test)

Uses XDataSource to create a connection using method getConnection().  
XDataSource uses XConnectionManager

## [calcite-test](./calcite-test)

Simple query test using Apache Calcite

## [polardbx-test](./polardbx-test)

A more complete example:
- ParseSQL.java / TestSchemaManager.java - Parse a query
- SimpleDbQueryApp.java - Executes a simple query from code against percona-server 
- SimpleServer.java - Complete server , client connect to a server send a SQL query that runs against percona-server with xprotocol plugin

Option to run:

1) Type for an SQL and loop over  
Create a connection using XConnectionManager and manager.getConnection

2) Build execution plan

# Code Breakdown

General

- [mysql-server](./docs/mysql-server.md)

- [xplugin-test](./docs/xplugin-test.md)

- [select-test](./docs/select-test.md)

- [calcite-test](./docs/calcite-test.md)


PolarDBX Test

- [ParseSQL.java](./docs/polardbx-test-parse.md)

- [SimpleDBQueryApp.java](./docs/polardbx-test-db-query.md)

- [SimpleServer.java](./docs/polardbx-test-server.md)

