Below is an **organized English version** of the documentation about **Druid’s SQL Parser** and **AST (Abstract Syntax Tree)**, translated and compiled from the provided Chinese documentation. This guide explains how the parser generates AST nodes, how to traverse and manipulate them, and the various classes and interfaces that compose Druid’s SQL Parser framework.

---

## 1. Overview: What is an AST?

**AST** stands for **Abstract Syntax Tree**. It is a tree representation of the syntactic structure of source code (in this case, SQL). Like many parsers, the Druid Parser converts SQL statements into an AST for further analysis, transformation, or processing.

---

## 2. AST Node Types in Druid

In Druid, most AST node classes derive from three main interfaces:

1. **`SQLObject`**  
2. **`SQLExpr`** (extends `SQLObject`)  
3. **`SQLStatement`** (extends `SQLObject`)

Additionally, there are specialized sub-interfaces such as **`SQLTableSource`** (also extending `SQLObject`) and classes like `SQLSelect`, `SQLSelectQueryBlock`, etc.

The hierarchy can be summarized as follows:

```java
package com.alibaba.druid.sql.ast;

public interface SQLObject {}

public interface SQLExpr extends SQLObject {}

public interface SQLStatement extends SQLObject {}

// Additional interfaces/classes in the AST
public interface SQLTableSource extends SQLObject {}

public class SQLSelect extends SQLObject {}
public class SQLSelectQueryBlock extends SQLObject {}
```

### 2.1 Common SQL Expression Classes (`SQLExpr`)

#### 2.1.1 `SQLName`
**`SQLName`** is a sub-interface of **`SQLExpr`**, used to represent identifiers and properties, such as table names, column names, or aliases. Two main implementations:

- **`SQLIdentifierExpr`**: Represents a simple identifier (e.g., `id`, `name`, etc.).
  
  ```java
  class SQLIdentifierExpr implements SQLExpr, SQLName {
      String name; 
      // e.g., "ID" in "SELECT ID FROM ..."
  }
  ```

- **`SQLPropertyExpr`**: Represents a qualified name with an owner (e.g., `A.ID`).
  
  ```java
  class SQLPropertyExpr implements SQLExpr, SQLName {
      SQLExpr owner;  // e.g., "A"
      String name;    // e.g., "ID"
  }
  ```

#### 2.1.2 `SQLBinaryOpExpr`
Represents a binary operation like `id = 3` or `age > 10`. The AST captures the operation type (`operator`) and the left/right sub-expressions:
```java
class SQLBinaryOpExpr implements SQLExpr {
    SQLExpr left;
    SQLExpr right;
    SQLBinaryOperator operator;
    // e.g., "ID = 3"
}
```

#### 2.1.3 `SQLVariantRefExpr`
Represents a parameter placeholder, often `?` or named parameters:
```java
class SQLVariantRefExpr extends SQLExprImpl {
    String name; // e.g., "?" for placeholders
}
```

#### 2.1.4 `SQLIntegerExpr`
Represents integer numeric literals:
```java
public class SQLIntegerExpr extends SQLNumericLiteralExpr implements SQLValuableExpr {
    Number number;

    @Override
    public Object getValue() {
        return this.number;
    }
}
```

#### 2.1.5 `SQLCharExpr`
Represents character (string) literals:
```java
public class SQLCharExpr extends SQLTextLiteralExpr implements SQLValuableExpr {
    String text; // e.g., the content of 'jobs'
}
```

---

### 2.2 Common SQL Statements (`SQLStatement`)

Druid’s most common `SQLStatement` classes are:

- **`SQLSelectStatement`**  
- **`SQLUpdateStatement`**  
- **`SQLDeleteStatement`**  
- **`SQLInsertStatement`**  

Example definitions:
```java
class SQLSelectStatement implements SQLStatement {
    SQLSelect select;
}

class SQLUpdateStatement implements SQLStatement {
    SQLExprTableSource tableSource;
    List<SQLUpdateSetItem> items;
    SQLExpr where;
}

class SQLDeleteStatement implements SQLStatement {
    SQLTableSource tableSource;
    SQLExpr where;
}

class SQLInsertStatement implements SQLStatement {
    SQLExprTableSource tableSource;
    List<SQLExpr> columns;
    SQLSelect query; 
}
```

---

### 2.3 SQL Table Sources

A **`SQLTableSource`** represents the `FROM` part of the SQL. It can be:

- **`SQLExprTableSource`**: A simple table reference.
- **`SQLJoinTableSource`**: A join between two table sources.
- **`SQLSubqueryTableSource`**: A sub-select as the table source.
- **`SQLWithSubqueryClause.Entry`**: A CTE (Common Table Expression) entry in a `WITH` clause.

Examples:

1. **`SQLExprTableSource`**  
   ```java
   class SQLExprTableSource extends SQLTableSourceImpl {
       SQLExpr expr; // Usually a SQLIdentifierExpr for "emp"
   }
   ```
   Used in a simple `SELECT * FROM emp`.

2. **`SQLJoinTableSource`**  
   ```java
   class SQLJoinTableSource extends SQLTableSourceImpl {
       SQLTableSource left;
       SQLTableSource right;
       JoinType joinType;  // e.g. INNER_JOIN, LEFT_OUTER_JOIN, etc.
       SQLExpr condition;  // e.g. e.org_id = o.id
   }
   ```
   Represents `SELECT * FROM emp e INNER JOIN org o ON e.org_id = o.id`.

3. **`SQLSubqueryTableSource`**  
   ```java
   class SQLSubqueryTableSource extends SQLTableSourceImpl {
       SQLSelect select;
   }
   ```
   Represents `SELECT * FROM (SELECT * FROM temp) a`.

4. **`SQLWithSubqueryClause.Entry`**  
   Represents a named subquery for use with `WITH ...` clauses.

---

### 2.4 SQLSelect and SQLSelectQuery

A `SQLSelectStatement` holds a **`SQLSelect`** object, which in turn holds a **`SQLSelectQuery`**. Two main implementations of `SQLSelectQuery` are:

1. **`SQLSelectQueryBlock`**
2. **`SQLUnionQuery`**

#### `SQLSelectQueryBlock`
Represents a standard `SELECT ... FROM ... WHERE ... GROUP BY ... ORDER BY ...` structure:
```java
class SQLSelectQueryBlock implements SQLSelectQuery {
    List<SQLSelectItem> selectList;
    SQLTableSource from;
    SQLExprTableSource into;
    SQLExpr where;
    SQLSelectGroupByClause groupBy;
    SQLOrderBy orderBy;
    SQLLimit limit;
}
```

#### `SQLUnionQuery`
Represents a `SELECT ... UNION SELECT ...` statement:
```java
class SQLUnionQuery implements SQLSelectQuery {
    SQLSelectQuery left;
    SQLSelectQuery right;
    SQLUnionOperator operator; // e.g. UNION, UNION_ALL, MINUS, INTERSECT
}
```

---

### 2.5 `SQLCreateTableStatement`
A DDL statement for creating a table, which can contain multiple definitions or constraints:

```java
public class SQLCreateTableStatement extends SQLStatementImpl
        implements SQLDDLStatement, SQLCreateStatement {
    SQLExprTableSource tableSource;
    List<SQLTableElement> tableElementList;
    SQLSelect select;
    
    // Utility methods inside:
    public SQLColumnDefinition findColumn(String columName) {}
    public SQLTableElement findIndex(String columnName) {}
    public boolean isReferenced(String tableName) {}
}
```

---

## 3. Generating the AST

### 3.1 Parsing SQL into a List of Statements
Use **`SQLUtils.parseStatements(sql, dbType)`** to parse SQL text into a list of `SQLStatement` objects:

```java
import com.alibaba.druid.util.JdbcConstants;

String dbType = JdbcConstants.MYSQL;
String sql = "SELECT * FROM emp WHERE id = 3";

List<SQLStatement> statementList = SQLUtils.parseStatements(sql, dbType);
// statementList will hold one or more SQLStatement objects
```

### 3.2 Parsing an Expression
Use **`SQLUtils.toSQLExpr(exprString, dbType)`** to parse a standalone SQL expression:

```java
String dbType = JdbcConstants.MYSQL;
SQLExpr expr = SQLUtils.toSQLExpr("id=3", dbType);
// expr is now an AST node representing "id=3"
```

---

## 4. Printing AST Nodes

Druid provides utility methods to convert AST nodes back to SQL strings:

```java
public class SQLUtils {
    // Convert a single SQLObject (SQLExpr or SQLStatement) to a String
    public static String toSQLString(SQLObject sqlObj, String dbType);

    // Convert multiple SQLStatements to a String
    public static String toSQLString(List<SQLStatement> statementList, String dbType);
}
```

These methods are helpful for formatting or logging after AST transformations.

---

## 5. Custom AST Traversal (Visitors)

**All AST nodes in Druid support the Visitor design pattern.** You can create a custom **Visitor** (or extend `ASTVisitorAdapter`) to traverse the AST and perform your desired logic.

Druid ships with various built-in visitors, for example:
- **`SchemaStatVisitor`**: Collects schema-related information (tables, columns, conditions).
- **`WallVisitor`**: SQL injection prevention logic.
- **`ParameterizedOutputVisitor`**: Used to parameterize SQL for merging.
- **`EvalVisitor`**: Used to evaluate expressions at runtime.
- **`ExportParameterVisitor`**: Extracts parameters from SQL.
- **`OutputVisitor`**: Prints out the AST as a SQL string.
  
A simple example of a custom Visitor:

```java
public static class ExportTableAliasVisitor extends MySqlASTVisitorAdapter {
    private Map<String, SQLTableSource> aliasMap = new HashMap<>();
    
    @Override
    public boolean visit(SQLExprTableSource x) {
        String alias = x.getAlias();
        aliasMap.put(alias, x);
        return true;
    }

    public Map<String, SQLTableSource> getAliasMap() {
        return aliasMap;
    }
}
```

**Usage**:
```java
final String dbType = JdbcConstants.MYSQL;
String sql = "SELECT * FROM mytable a WHERE a.id = 3";
List<SQLStatement> stmtList = SQLUtils.parseStatements(sql, dbType);

ExportTableAliasVisitor visitor = new ExportTableAliasVisitor();
for (SQLStatement stmt : stmtList) {
    stmt.accept(visitor);
}

SQLTableSource tableSource = visitor.getAliasMap().get("a");
System.out.println(tableSource); // prints "mytable a" or the AST node details
```

---

## 6. SQL Parser Overview

### 6.1 Introduction
Druid’s SQL Parser is used internally for:
- **SQL Injection Protection** (`WallFilter`)
- **SQL Merge/Normalization** (in `StatFilter`)
- **SQL Formatting**
- **Sharding** (splitting across multiple databases/tables)

### 6.2 Comparison with ANTLR
Unlike ANTLR-generated parsers, Druid’s Parser is *handwritten*, focusing on **performance** and **production readiness**. It can parse and analyze SQL statements extremely quickly, making it suitable for real-time tasks like injection prevention or query rewriting.

### 6.3 Supported Dialects
- **MySQL** (fully supported)
- **ODPS/Hive** (fully supported for ODPS, commonly used Hive syntax supported)
- **PostgreSQL** (fully supported)
- **Oracle** (most commonly used features supported)
- **SQL Server** (common features supported)
- **DB2** (common features)
- … and partial support for other SQL-92-like dialects

### 6.4 Performance
Druid’s parser can parse a simple SQL such as `SELECT ID, NAME, AGE FROM USER WHERE ID = ?` in about **600 nanoseconds** on a modern CPU, which can translate to millions of parses per second in a single thread.

### 6.5 Code Structure
- **Lexer**: Tokenizes the input SQL string.
- **Parser**: Converts tokens to an AST.
- **AST**: The Abstract Syntax Tree (detailed above).
- **Visitor**: Processes AST nodes.

### 6.6 Dialects
While much of SQL is standard, each database has its own peculiarities. Druid has specific parser variants (and visitors) for MySQL, Oracle, PostgreSQL, SQL Server, ODPS, etc.

---

## 7. Schema Repository

Druid’s `SchemaRepository` allows you to maintain an in-memory schema (metadata) for tables, columns, etc., enabling column resolution and more advanced semantics.  

Example usage:
```java
import com.alibaba.druid.sql.repository.SchemaRepository;
import com.alibaba.druid.util.JdbcConstants;

String dbType = JdbcConstants.MYSQL;
SchemaRepository repository = new SchemaRepository(dbType);

// Provide commands to simulate a console:
repository.console("create table t_emp(emp_id bigint, name varchar(20));");
repository.console("create table t_org(org_id bigint, name varchar(20));");

String sql = "SELECT emp_id, a.name AS emp_name, org_id, b.name AS org_name\n" +
             "FROM t_emp a INNER JOIN t_org b ON a.emp_id = b.org_id";

List<SQLStatement> stmtList = SQLUtils.parseStatements(sql, dbType);
SQLSelectStatement selectStmt = (SQLSelectStatement) stmtList.get(0);

// Use the repository to resolve columns
repository.resolve(selectStmt);

// Now the statement's columns can be resolved to their tables/definitions
SQLSelectQueryBlock queryBlock = selectStmt.getSelect().getQueryBlock();
SQLSelectItem item = queryBlock.findSelectItem("org_name");
// item.getExpr() -> SQLPropertyExpr "b.name" 
// With repository.resolve, item.getExpr().getResolvedColumn() gives the definition from t_org.name
```

You can also use:
- `repository.console("show columns from tableName")` to list columns as if in a MySQL console.
- DDL statements like `ALTER TABLE t_emp ADD COLUMN ...` to update the in-memory schema.

---

## 8. Additional Utilities

### 8.1 Evaluating SQL Expressions (`EvalVisitor`)

Druid includes `EvalVisitor`, which can evaluate SQL expressions at runtime. For convenience, there’s a static helper **`SQLEvalVisitorUtils.evalExpr(dbType, expr, parameters...)`**:

```java
Object value = SQLEvalVisitorUtils.evalExpr(JdbcConstants.MYSQL, "3+4");
// value => 7

boolean inTest = (Boolean) SQLEvalVisitorUtils.evalExpr(JdbcConstants.MYSQL, "? IN (1, 2, 3)", 2);
// inTest => true if the parameter matches
```

### 8.2 Collecting Schema Stats (`SchemaStatVisitor`)

The `SchemaStatVisitor` collects information such as which tables and columns are accessed:

```java
String sql = "SELECT name, age FROM t_user WHERE id = 1";
String dbType = JdbcConstants.MYSQL;

List<SQLStatement> stmtList = SQLUtils.parseStatements(sql, dbType);
SQLStatement stmt = stmtList.get(0);

SchemaStatVisitor statVisitor = SQLUtils.createSchemaStatVisitor(dbType);
stmt.accept(statVisitor);

System.out.println(statVisitor.getTables());   // {t_user=Select}
System.out.println(statVisitor.getColumns());  // [t_user.name, t_user.age, t_user.id]
System.out.println(statVisitor.getConditions()); // [t_user.id = 1]
```

### 8.3 SQL Formatting

Druid offers a **semantic SQL formatter** via **`SQLUtils.format(sql, dbType)`**. For example:

```java
String sql = "update t set name = 'x' where id < 100 limit 10";
String formatted = SQLUtils.format(sql, JdbcConstants.MYSQL);
System.out.println(formatted);  
```

The default output (uppercase keywords) might be:

```sql
UPDATE t
SET name = 'x'
WHERE id < 100
LIMIT 10
```

You can also specify a lowercase formatting option with `SQLUtils.DEFAULT_LCASE_FORMAT_OPTION`.

---

## 9. References

- [Druid SQL Parser (Official GitHub Wiki)](https://github.com/alibaba/druid/wiki/SQL-Parser)  
- [Druid SQL AST Reference](https://github.com/alibaba/druid/wiki/Druid_SQL_AST)  
- [SQL Schema Repository](https://github.com/alibaba/druid/wiki/SQL_Schema_Repository)  
- [SQL Parser Visitor Demo](https://github.com/alibaba/druid/wiki/SQL_Parser_Demo_visitor)  
- [Druid GitHub Repository](https://github.com/alibaba/druid)

---

### Conclusion

Using Druid’s **SQL Parser**, you can:

- Parse SQL statements or individual expressions into an AST.
- Traverse or manipulate the AST with **Visitors**.
- Reconstruct SQL from the AST.
- Maintain an in-memory schema repository for advanced column resolution.
- Evaluate expressions at runtime.
- Format SQL queries with a semantically-aware formatter.

