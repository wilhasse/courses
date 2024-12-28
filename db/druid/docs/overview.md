Below is an in-depth overview of **Alibaba Druid**—specifically focusing on its **SQL Parser** (often called “FastSQL Parser”) and how you can use it for **MySQL** queries. We’ll go through its **key concepts**, **typical usage**, and **AST (Abstract Syntax Tree) manipulation**.

> **Note:**
>  Alibaba Druid is sometimes confused with **Apache Druid** (the real-time OLAP system). Alibaba Druid is **not** related to Apache Druid. Alibaba Druid is a Java-based library that includes:
>
> 1. A high-performance JDBC connection pool (like HikariCP or C3P0).
> 2. A robust SQL parser capable of handling multiple dialects (MySQL, Oracle, PostgreSQL, etc.).
> 3. SQL analysis tools (e.g., SQL stat and monitoring).

Below, we focus on **the parser** portion.

------

## 1. What Is Alibaba Druid’s SQL Parser?

Alibaba Druid’s parser:

1. **Tokenizes** your SQL string (lexical analysis).
2. **Parses** it according to the grammar rules of a particular dialect (MySQL, Oracle, PostgreSQL, etc.).
3. Produces an **AST** made up of classes in `com.alibaba.druid.sql.ast.*`.

Alibaba Druid is especially **MySQL-friendly** and has good coverage of MySQL-specific syntax (including hints, partition DDL, MySQL-specific expressions, etc.).

------

## 2. Key Concepts & Architecture

### 2.1 Parsing Dialects

Druid supports multiple SQL dialects via different parser classes. For **MySQL**:

```java
import com.alibaba.druid.sql.dialect.mysql.parser.MySqlStatementParser;
```

Under the hood, MySqlStatementParser extends a common parser framework but includes MySQL-specific grammar.

### 2.2 The Parsing Process

1. **Lexical Analysis**
    The parser first **tokenizes** the SQL string into tokens like `SELECT`, `FROM`, `WHERE`, `IDENTIFIER`, `STRING_LITERAL`, etc.
   - Implemented by classes under `com.alibaba.druid.sql.parser.Lexer` (or specifically `MySqlLexer` for MySQL).
2. **Grammar Parsing**
    Next, it parses tokens using a **recursive descent parser**.
   - The grammar rules define how tokens form valid SQL statements: `SELECT ... FROM ... WHERE ...`, etc.
   - This step builds an AST with classes like `SQLSelectStatement`, `SQLInsertStatement`, `SQLAlterTableStatement`, etc.
3. **AST Representation**
    After the parser finishes, you get a list of `SQLStatement` objects. Each statement is a tree of child nodes (e.g., a `SQLSelectQueryBlock` for the `FROM`, `WHERE`, `ORDER BY`, etc.).

------

## 3. Typical Usage: Parsing a MySQL Query

A minimal example for MySQL:

```java
import com.alibaba.druid.sql.ast.SQLStatement;
import com.alibaba.druid.sql.dialect.mysql.parser.MySqlStatementParser;

String sql = "SELECT col2 FROM table1 WHERE col1 > 100 ORDER BY col2";

MySqlStatementParser parser = new MySqlStatementParser(sql);
List<SQLStatement> statementList = parser.parseStatementList();

// Usually just one statement, but could be multiple if the SQL has multiple semicolon-separated statements
SQLStatement stmt = statementList.get(0);

// We often cast it if we know it's a SELECT
// e.g. SQLSelectStatement selectStmt = (SQLSelectStatement) stmt;
```

After this, `stmt` is a **tree** representing the SQL.

------

## 4. AST Structure & Common Classes

Below are some frequently encountered classes in Alibaba Druid’s AST:

1. **`SQLSelectStatement`**
   - Represents a `SELECT ...` statement.
   - Has a `SQLSelect` object, which holds a `SQLSelectQueryBlock` for the core query.
2. **`SQLSelectQueryBlock`**
   - Represents the main query block: `SELECT [selectList] FROM [table] WHERE [condition] GROUP BY ... ORDER BY ...`
   - Fields like `.getSelectList()`, `.getFrom()`, `.getWhere()`, `.getOrderBy()`, etc.
3. **`SQLTableSource`**
   - Represents `FROM table1` or `JOIN ...` structures.
   - For a simple table reference, often a `SQLExprTableSource` with `.getExpr() = SQLIdentifierExpr(tableName)`.
4. **`SQLExpr` (Expressions)**
   - For conditions, columns, or function calls.
   - For example, a `SQLBinaryOpExpr` might represent `col1 > 100`.
   - `SQLVariantRefExpr` might represent a `?` parameter or `:paramName`.
5. **`SQLOrderBy` / `SQLSelectOrderByItem`**
   - Represent `ORDER BY` clauses, including ASC/DESC.

Alibaba Druid also has MySQL-specific classes (like `MySqlSelectQueryBlock`) if you’re dealing with MySQL-only syntax or features.

------

## 5. How to Inspect / Manipulate the AST

### 5.1 Direct Accessors & Mutators

After parsing, you can **traverse** the AST by calling getters and setters. For instance:

```java
SQLSelectStatement selectStmt = (SQLSelectStatement) stmt;
SQLSelectQueryBlock queryBlock = (SQLSelectQueryBlock) selectStmt.getSelect().getQuery();

// FROM
SQLTableSource from = queryBlock.getFrom();

// WHERE
SQLExpr where = queryBlock.getWhere();
if (where instanceof SQLBinaryOpExpr) {
    SQLBinaryOpExpr boe = (SQLBinaryOpExpr) where;
    // boe.getOperator() => SQLBinaryOperator.GreaterThan, etc.
    // boe.getLeft() / boe.getRight()
}

// ORDER BY
SQLOrderBy orderBy = queryBlock.getOrderBy();
```

You can **modify** these objects to rewrite the SQL. For example, you can replace the `WHERE` clause with a new `SQLBinaryOpExpr` or add a condition.

### 5.2 Visitor Pattern

Druid also supports a **visitor pattern** (`SQLASTVisitor` or specialized dialect visitors like `MySqlASTVisitor`), so you can walk the AST generically. For example:

```java
import com.alibaba.druid.sql.visitor.SQLASTVisitor;

public class MyVisitor extends SQLASTVisitorAdapter {
    @Override
    public boolean visit(SQLSelectQueryBlock x) {
        // e.g. examine x.getWhere()
        return true; // returning true = continue visiting children
    }

    @Override
    public boolean visit(SQLBinaryOpExpr x) {
        // handle col1 > 100
        return true;
    }
}

// usage:
stmt.accept(new MyVisitor());
```

Visitors are handy if you need to systematically analyze or transform large/complex queries.

------

## 6. Converting AST Back to SQL String

After manipulation, you can convert the AST **back** to a SQL string via `SQLUtils.toMySqlString(...)`. For example:

```java
import com.alibaba.druid.sql.SQLUtils;
import com.alibaba.druid.util.JdbcConstants;

String newSql = SQLUtils.toSQLString(stmt, JdbcConstants.MYSQL);
// or
String newSql = SQLUtils.toMySqlString(stmt);
```

This is very convenient for rewriting queries.

------

## 7. Practical Use Cases

1. **SQL Rewriting / Transformation**
   - Insert “hints” or partition conditions automatically.
   - E.g., rewriting `SELECT * FROM table1` into multiple chunked queries with appended `WHERE col2 < X` conditions, etc.
2. **SQL Validation**
   - You can parse a query, ensure it meets certain rules (e.g., no subqueries, only certain columns used).
3. **SQL Analysis** (similar to what you might do with a query plan)
   - Quickly identify `WHERE` conditions, `GROUP BY`, `ORDER BY`, etc.
   - Then decide how to route or parallelize queries.
4. **SQL Security / Auditing**
   - Check if the query tries to do a destructive operation, or if it’s reading from unauthorized tables.

------

## 8. Performance & Popularity

- Druid’s parser is known to be **fast** (often called “FastSQL”), widely used inside Alibaba, and stable for large-scale online scenarios.
- It’s popular on GitHub: [Alibaba Druid on GitHub](https://github.com/alibaba/druid). You’ll find many examples, though the official documentation can be a bit scattered.
- Many people primarily use Druid as a **connection pool** with built-in SQL stat monitoring, but it also has powerful parsing features that you can use standalone.

------

## 9. Helpful Tips for Using the Parser

1. **Add Just the Parser Dependency**
    You do **not** have to use the entire Druid connection pool. If you only need the parser, you can reference `com.alibaba:druid` and just use the parser classes.
2. **Stay Up-to-Date**
    Different releases support more MySQL features. If you see parse errors on new MySQL syntax, check if a newer Druid version might fix it.
3. **Handling Edge Cases**
    MySQL has many quirks (e.g., multi-statements, exotic functions, hints, etc.). If you run into parse issues, try the latest version or open an issue in the GitHub repo.
4. **AST vs. Raw String**
    Sometimes it’s tempting to do string manipulations, but you’ll have more robust results if you modify the AST.
5. **Test with Complex Queries**
    If you plan on rewriting subqueries, joins, or window functions, thoroughly test them—some advanced syntax might have partial coverage, though it’s quite comprehensive for most mainstream use cases.

------

## 10. Summary

- **Alibaba Druid’s SQL Parser** is a **powerful, MySQL-friendly** solution for parsing and rewriting queries in Java.
- You can parse a SQL string into an AST (`SQLStatement` objects), **inspect** or **modify** the AST, and **re-generate** the updated SQL string.
- Common usage includes **query rewriting**, **analysis**, **sharding logic**, or **custom optimization** (as in PolarDBX, which uses it to front-end the Calcite optimizer).
- While documentation can be sparse, you’ll find usage examples in the [Alibaba Druid GitHub repo](https://github.com/alibaba/druid) and by examining open-source projects like PolarDBX (which rely on it extensively).

That’s the big picture of how to parse MySQL queries with **Alibaba Druid**. If you need deeper examples, you can explore **PolarDBX** or look for other open-source projects that use Druid to see more advanced usage patterns.