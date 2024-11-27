# Preparation

Plugin enable

```bash
show plugins;
| mysqlx                           | ACTIVE   | DAEMON             | NULL          | GPL     |
```

Listen default port 33060

```bash
cat mysql-error.log
2024-11-27T02:28:27.275744Z 0 [Note] [MY-011240] [Server] Plugin mysqlx reported: 'Using SSL configuration from MySQL Server'
2024-11-27T02:28:27.278335Z 0 [Note] [MY-011243] [Server] Plugin mysqlx reported: 'Using OpenSSL for TLS connections'
2024-11-27T02:28:27.278711Z 0 [System] [MY-011323] [Server] X Plugin ready for connections. Bind-address: '::' port: 33060, socket: /tmp/mysqlx.sock
```

Test bind listening to all interfaces

```bash
mysql> SHOW VARIABLES LIKE 'mysqlx_bind_address';
+---------------------+-------+
| Variable_name       | Value |
+---------------------+-------+
| mysqlx_bind_address | *     |
+---------------------+-------+
1 row in set (0,01 sec)
```

Authentication

```bash
CREATE USER 'teste'@'10.1.1.158'
GRANT ALL PRIVILEGES ON *.* TO 'teste'@'10.1.1.158';
SET PASSWORD FOR 'teste'@'10.1.1.158' = 'teste';
```

# Login

New protocol , access denied switch to old protocol

```bash
./mysqlxtest -u teste -p teste --trace-protocol -h 10.1.1.158 -P 3306
<<<< RECEIVE 5 Mysqlx.Notice.Frame {
  type: 5
  payload: ""
}
<<<< RECEIVE 1 Mysqlx.Ok {
}
>>>> SEND 16 Mysqlx.Session.AuthenticateStart {
  mech_name: "SHA256_MEMORY"
}
<<<< RECEIVE 23 Mysqlx.Session.AuthenticateContinue {
  auth_data: "\t\020%\001[AK0+\026*%S\006\016\005%R\\\000"
}
>>>> SEND 74 Mysqlx.Session.AuthenticateContinue {
  auth_data: "\000teste\00000DD3A58BCE3EFF08009A5BDB9B61B8F4EE0D3741F33A683A3BE369C8911A5D7"
}
<<<< RECEIVE 80 Mysqlx.Error {
  severity: ERROR
  code: 1045
  msg: "Access denied for user \'teste\'@\'10.1.1.158\' (using password: YES)"
  sql_state: "HY000"
}
>>>> SEND 10 Mysqlx.Session.AuthenticateStart {
  mech_name: "MYSQL41"
}
<<<< RECEIVE 23 Mysqlx.Session.AuthenticateContinue {
  auth_data: "v\032%Xv\006JAlF\017\0219n<]zD9\000"
}
>>>> SEND 51 Mysqlx.Session.AuthenticateContinue {
  auth_data: "\000teste\000*82D9E5FD9AA52188E3A81FEB36B916C40E1ECB70"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: CLIENT_ID_ASSIGNED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 26\n}\n }"
}
<<<< RECEIVE 3 Mysqlx.Session.AuthenticateOk {
  auth_data: ""
}
```

Using old protocol goes directly and I typed SELECT 1; with success

```bash
./mysqlxtest -u teste -p teste --trace-protocol --mysql41-auth -h 10.1.1.158 -P 33060
<<<< RECEIVE 5 Mysqlx.Notice.Frame {
  type: 5
  payload: ""
}
<<<< RECEIVE 1 Mysqlx.Ok {
}
>>>> SEND 10 Mysqlx.Session.AuthenticateStart {
  mech_name: "MYSQL41"
}
<<<< RECEIVE 23 Mysqlx.Session.AuthenticateContinue {
  auth_data: "MD&7\021((jwdN\"\034dEkJVn\000"
}
>>>> SEND 51 Mysqlx.Session.AuthenticateContinue {
  auth_data: "\000teste\000*74A03EEA72DEC60436A56EAA0BC9C5D069288151"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: CLIENT_ID_ASSIGNED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 25\n}\n }"
}
<<<< RECEIVE 3 Mysqlx.Session.AuthenticateOk {
  auth_data: ""
}
SELECT 1;
RUN SELECT 1
>>>> SEND 11 Mysqlx.Sql.StmtExecute {
  stmt: "SELECT 1"
}
<<<< RECEIVE 23 Mysqlx.Resultset.ColumnMetaData {
  type: SINT
  name: "1"
  original_name: ""
  table: ""
  original_table: ""
  schema: ""
  catalog: "def"
  length: 2
  flags: 16
}
<<<< RECEIVE 4 Mysqlx.Resultset.Row {
  field: "\002"
}
<<<< RECEIVE 1 Mysqlx.Resultset.FetchDone {
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}
1
1
0 rows affected
quit; 
```

# Run a file

Create test database

```bash
mysqladmin create test -u root -p -h 10.1.1.158
```

Now run a simple test file

```bash
-->sql
USE test;
-->endsql
-->sql
CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name VARCHAR(50), age INT);
-->endsql
-->echo Inserting test data
-->sql
INSERT INTO users VALUES (1, 'John', 25), (2, 'Jane', 30);
-->endsql
-->echo Running SELECT queries
-->sql
SELECT * FROM users;
-->endsql
-->sql
SELECT name FROM users WHERE age > 25;
-->endsql
-->echo Testing document store functionality
-->sql
CREATE TABLE IF NOT EXISTS docs (doc JSON);
-->endsql
-->echo Inserting JSON documents
-->sql
INSERT INTO docs (doc) VALUES ('{"name": "Alice", "hobbies": ["reading", "hiking"]}');
-->endsql
-->sql
INSERT INTO docs (doc) VALUES ('{"name": "Bob", "hobbies": ["gaming", "cooking"]}');
-->endsql
-->echo Querying JSON data
-->sql
SELECT doc->>'$.name' as name, JSON_EXTRACT(doc, '$.hobbies') as hobbies FROM docs;
-->endsql
-->echo Cleanup
-->sql
DROP TABLE users;
-->endsql
-->sql
DROP TABLE docs;
-->endsql
```

Result:

```bash
./mysqlxtest -u teste -p teste --trace-protocol --mysql41-auth -h 10.1.1.158 -P 33060 --file=test.txt

RUN USE test
>>>> SEND 11 Mysqlx.Sql.StmtExecute {
  stmt: "USE test"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

0 rows affected
RUN CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name VARCHAR(50), age INT)
>>>> SEND 83 Mysqlx.Sql.StmtExecute {
  stmt: "CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name VARCHAR(50), age INT)"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

0 rows affected
Inserting test data
RUN INSERT INTO users VALUES (1, 'John', 25), (2, 'Jane', 30)
>>>> SEND 60 Mysqlx.Sql.StmtExecute {
  stmt: "INSERT INTO users VALUES (1, \'John\', 25), (2, \'Jane\', 30)"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 2\n}\n }"
}
<<<< RECEIVE 61 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: PRODUCED_MESSAGE\nvalue {\n  type: V_STRING\n  v_string {\n    value: \"Records: 2  Duplicates: 0  Warnings: 0\"\n  }\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

2 rows affected
Records: 2  Duplicates: 0  Warnings: 0
Running SELECT queries
RUN SELECT * FROM users
>>>> SEND 22 Mysqlx.Sql.StmtExecute {
  stmt: "SELECT * FROM users"
}
<<<< RECEIVE 40 Mysqlx.Resultset.ColumnMetaData {
  type: SINT
  name: "id"
  original_name: "id"
  table: "users"
  original_table: "users"
  schema: "test"
  catalog: "def"
  length: 11
  flags: 48
}
<<<< RECEIVE 46 Mysqlx.Resultset.ColumnMetaData {
  type: BYTES
  name: "name"
  original_name: "name"
  table: "users"
  original_table: "users"
  schema: "test"
  catalog: "def"
  collation: 255
  length: 200
}
<<<< RECEIVE 40 Mysqlx.Resultset.ColumnMetaData {
  type: SINT
  name: "age"
  original_name: "age"
  table: "users"
  original_table: "users"
  schema: "test"
  catalog: "def"
  length: 11
}
<<<< RECEIVE 14 Mysqlx.Resultset.Row {
  field: "\002"
  field: "John\000"
  field: "2"
}
<<<< RECEIVE 14 Mysqlx.Resultset.Row {
  field: "\004"
  field: "Jane\000"
  field: "<"
}
<<<< RECEIVE 1 Mysqlx.Resultset.FetchDone {
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}
id      name    age
1       John    25
2       Jane    30
0 rows affected
RUN SELECT name FROM users WHERE age > 25
>>>> SEND 40 Mysqlx.Sql.StmtExecute {
  stmt: "SELECT name FROM users WHERE age > 25"
}
<<<< RECEIVE 46 Mysqlx.Resultset.ColumnMetaData {
  type: BYTES
  name: "name"
  original_name: "name"
  table: "users"
  original_table: "users"
  schema: "test"
  catalog: "def"
  collation: 255
  length: 200
}
<<<< RECEIVE 8 Mysqlx.Resultset.Row {
  field: "Jane\000"
}
<<<< RECEIVE 1 Mysqlx.Resultset.FetchDone {
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}
name
Jane
0 rows affected
Testing document store functionality
RUN CREATE TABLE IF NOT EXISTS docs (doc JSON)
>>>> SEND 45 Mysqlx.Sql.StmtExecute {
  stmt: "CREATE TABLE IF NOT EXISTS docs (doc JSON)"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

0 rows affected
Inserting JSON documents
RUN INSERT INTO docs (doc) VALUES ('{"name": "Alice", "hobbies": ["reading", "hiking"]}')
>>>> SEND 88 Mysqlx.Sql.StmtExecute {
  stmt: "INSERT INTO docs (doc) VALUES (\'{\"name\": \"Alice\", \"hobbies\": [\"reading\", \"hiking\"]}\')"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 1\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

1 rows affected
RUN INSERT INTO docs (doc) VALUES ('{"name": "Bob", "hobbies": ["gaming", "cooking"]}')
>>>> SEND 86 Mysqlx.Sql.StmtExecute {
  stmt: "INSERT INTO docs (doc) VALUES (\'{\"name\": \"Bob\", \"hobbies\": [\"gaming\", \"cooking\"]}\')"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 1\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

1 rows affected
Querying JSON data
RUN SELECT doc->>'$.name' as name, JSON_EXTRACT(doc, '$.hobbies') as hobbies FROM docs
>>>> SEND 85 Mysqlx.Sql.StmtExecute {
  stmt: "SELECT doc->>\'$.name\' as name, JSON_EXTRACT(doc, \'$.hobbies\') as hobbies FROM docs"
}
<<<< RECEIVE 30 Mysqlx.Resultset.ColumnMetaData {
  type: BYTES
  name: "name"
  original_name: ""
  table: ""
  original_table: ""
  schema: ""
  catalog: "def"
  collation: 46
  length: 4294967292
}
<<<< RECEIVE 35 Mysqlx.Resultset.ColumnMetaData {
  type: BYTES
  name: "hobbies"
  original_name: ""
  table: ""
  original_table: ""
  schema: ""
  catalog: "def"
  collation: 46
  length: 4294967295
  content_type: 2
}
<<<< RECEIVE 33 Mysqlx.Resultset.Row {
  field: "Alice\000"
  field: "[\"reading\", \"hiking\"]\000"
}
<<<< RECEIVE 31 Mysqlx.Resultset.Row {
  field: "Bob\000"
  field: "[\"gaming\", \"cooking\"]\000"
}
<<<< RECEIVE 1 Mysqlx.Resultset.FetchDone {
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}
name    hobbies
Alice   ["reading", "hiking"]
Bob     ["gaming", "cooking"]
0 rows affected
Cleanup
RUN DROP TABLE users
>>>> SEND 19 Mysqlx.Sql.StmtExecute {
  stmt: "DROP TABLE users"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

0 rows affected
RUN DROP TABLE docs
>>>> SEND 18 Mysqlx.Sql.StmtExecute {
  stmt: "DROP TABLE docs"
}
<<<< RECEIVE 15 Mysqlx.Notice.Frame {
  type: 3
  scope: LOCAL
  payload: "Mysqlx.Notice.SessionStateChanged { param: ROWS_AFFECTED\nvalue {\n  type: V_UINT\n  v_unsigned_int: 0\n}\n }"
}
<<<< RECEIVE 1 Mysqlx.Sql.StmtExecuteOk {
}

0 rows affected
>>>> SEND 1 Mysqlx.Connection.Close {
}
<<<< RECEIVE 7 Mysqlx.Ok {
  msg: "bye!"
}
Mysqlx.Ok {
  msg: "bye!"
}
ok
```