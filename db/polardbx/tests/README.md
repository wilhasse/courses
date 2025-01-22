# Projects

## [mysql-server](./mysql-server)

Simple mysql server that respond to SELECT 1 implementing wire protocol

## [xplugin-test](./xplugin-test)

Testing get a list of databases using native MySQL X Plugin

## [select-test](./select-test)

Testing a select interacting to PolarDBX Engine XProtocol
Uses XDataSource to create a connection using method getConnection().  
XDataSource uses XConnectionManager

## [calcite-test](./calcite-test)

Simple query breakdown test using Apache Calcite Framework

## [polardbx-test](./polardbx-test)

A more complete example:
- QuerySQL.java - Executes a simple query from code against percona-server 
- ParseSQL.java / TestSchemaManager.java - Parse a query
- SimpleServer.java - Complete server , client connect to a server send a SQL query that runs against percona-server with xprotocol plugin
- SimpleSplitServer.java - Extends SimpleServer by spliting query to run by chunk

Option to run:

1) Type for an SQL and loop over  
Create a connection using XConnectionManager and manager.getConnection

2) Build execution plan

## [server-test](./server-test)

Final server test only with splitting query logic:
- Removed Extra Inheritance
- Moved Inline Classes to Helper Files
- Simplified Query Splitting Logic
- Refactoring for a Single Purpose

# General

[Code Documentation](./code_doc.md)
