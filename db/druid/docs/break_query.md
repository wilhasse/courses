Below is a step-by-step guide (with code examples) showing how you can use the Alibaba **Druid** SQL Parser (not to be confused with Apache Druid) to parse complex SQL into an Abstract Syntax Tree (AST), inspect or modify it, and then (if desired) generate new SQL or route parts of the query to different data sources such as Apache Doris or MySQL.

---

## 1. Basic Idea

1. **Parse the SQL** into Druid’s internal AST data structures.
2. **Analyze or transform** the AST:
   - Identify subqueries
   - Identify large tables vs small tables
   - Extract or modify conditions
   - Possibly split the query into multiple parts
3. **Rewrite or extract** subqueries/tables that need to be executed elsewhere (e.g., Apache Doris).
4. **Generate new SQL** from the transformed AST for each target data source, or run parts in the original MySQL.
5. **Combine results** in your application or a middle layer.

By using the AST, you have complete control over the query structure rather than doing string-based manipulations.

---

## 2. Adding the Alibaba Druid Dependency

If you are using Maven, the Alibaba Druid parser is typically brought in by:

```xml
<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>druid</artifactId>
    <version>1.2.16</version> <!-- Or a newer stable version -->
</dependency>
```

Make sure you have it on your classpath so you can use the parser, AST classes, and visitors.

---

## 3. Parsing SQL into an AST

### 3.1. Single Statement

```java
import com.alibaba.druid.sql.SQLUtils;
import com.alibaba.druid.sql.ast.SQLStatement;
import com.alibaba.druid.sql.ast.statement.SQLSelectStatement;
import com.alibaba.druid.util.JdbcConstants;

public class DruidParseDemo {

    public static void main(String[] args) {
        // Suppose you have a complex SQL with joins and subqueries:
        String sql = 
            "SELECT e.name, e.salary, o.name AS orgName\n" +
            "FROM employee e\n" +
            "     JOIN org o ON e.org_id = o.org_id\n" +
            "WHERE e.salary > 1000\n" +
            "ORDER BY e.salary";

        // dbType can be JdbcConstants.MYSQL, JdbcConstants.ORACLE, etc.
        String dbType = JdbcConstants.MYSQL;

        // Parse statements
        // parseStatements(...) returns a List<SQLStatement> in case multiple statements appear
        java.util.List<SQLStatement> stmtList = SQLUtils.parseStatements(sql, dbType);

        // For a single query, typically there's just one statement
        SQLStatement statement = stmtList.get(0);

        // If it is a SELECT, cast it to SQLSelectStatement
        if (statement instanceof SQLSelectStatement) {
            SQLSelectStatement selectStmt = (SQLSelectStatement) statement;

            // Now you have the AST for the entire SELECT
            // E.g. we can print it back out in a standardized format:
            String formatted = SQLUtils.toSQLString(selectStmt, dbType);
            System.out.println("Formatted SQL:\n" + formatted);
        }
    }
}
```

- `SQLUtils.parseStatements` does both lexical and syntactical analysis, returning a list of AST nodes (`SQLStatement`).
- If you have multiple statements separated by semicolons, each statement will be in that list.

### 3.2. Multiple Statements

If your input string contains multiple SQL statements (separated by `;`), Druid will return each one in order:

```java
String multiSql = "SELECT * FROM tableA; SELECT * FROM tableB WHERE id = 3;";
List<SQLStatement> stmtList = SQLUtils.parseStatements(multiSql, JdbcConstants.MYSQL);
System.out.println("Number of statements: " + stmtList.size());  // 2
```

---

## 4. Exploring the AST

Once you have a `SQLStatement`, you can drill down into its structure. The Druid AST classes you’ll most often encounter for `SELECT` statements are:

- **`SQLSelectStatement`**: top-level SELECT statement node
  - **`SQLSelect`** (its child) 
    - **`SQLSelectQueryBlock`** or **`SQLUnionQuery`**  
      - `from` (a `SQLTableSource`—could be a table, a join, or a subquery)
      - `selectList` (the projection items: columns, expressions, etc.)
      - `where`
      - `groupBy`
      - `orderBy`
      - `limit`
      - etc.

### 4.1 Navigating the AST

For a single SELECT without UNION, you typically deal with `SQLSelectQueryBlock`. For example:

```java
SQLSelectStatement selectStmt = (SQLSelectStatement) statement;

// The core of the query
SQLSelectQueryBlock queryBlock = selectStmt.getSelect().getQueryBlock();
if (queryBlock != null) {

    // 1) FROM clause
    SQLTableSource from = queryBlock.getFrom();
    System.out.println("FROM clause: " + from);

    // 2) SELECT items
    List<SQLSelectItem> selectItems = queryBlock.getSelectList();
    for (SQLSelectItem item : selectItems) {
        System.out.println("Select item: " + item.toString());
    }

    // 3) WHERE
    SQLExpr whereExpr = queryBlock.getWhere();
    System.out.println("WHERE expr: " + (whereExpr == null ? "none" : whereExpr.toString()));

    // 4) ORDER BY
    SQLOrderBy orderBy = queryBlock.getOrderBy();
    if (orderBy != null) {
        System.out.println("ORDER BY: " + orderBy.toString());
    }
}
```

### 4.2 `SQLTableSource` Subclasses

- **`SQLExprTableSource`**: A simple table reference, e.g. `employee e`.
- **`SQLJoinTableSource`**: A join of two table sources, e.g. `employee e JOIN org o ON e.org_id = o.org_id`.
- **`SQLSubqueryTableSource`**: A table source that itself is a subselect, e.g. `(SELECT * FROM ...) alias`.
- **`SQLUnionQuery`**: For handling `UNION`, `UNION ALL`, `INTERSECT`, etc.

When you have multiple joins, Druid typically creates a tree of `SQLJoinTableSource`s:

```java
SQLTableSource from = queryBlock.getFrom();
if (from instanceof SQLJoinTableSource) {
    SQLJoinTableSource join = (SQLJoinTableSource) from;
    SQLTableSource leftTable = join.getLeft();
    SQLTableSource rightTable = join.getRight();
    SQLExpr condition = join.getCondition();

    System.out.println("Join type: " + join.getJoinType());   // e.g. INNER_JOIN
    System.out.println("Left table: " + leftTable);
    System.out.println("Right table: " + rightTable);
    System.out.println("Join condition: " + condition);
}
```

For **subqueries**:

```java
if (from instanceof SQLSubqueryTableSource) {
    SQLSubqueryTableSource subQuery = (SQLSubqueryTableSource) from;
    SQLSelect subSelect = subQuery.getSelect();
    // Now subSelect has its own queryBlock or unionQuery
}
```

---

## 5. Rewriting or Splitting the Query

Because the AST is mutable, you can modify pieces of it or split it. A common scenario is:

1. Detect a “big” table reference in your JOIN or subquery.
2. Extract that portion of the query to run in a faster OLAP engine (e.g., Doris).
3. Keep the rest in MySQL or some other engine.
4. Re-combine data in the application layer.

**Example**: Suppose you have

```sql
SELECT a.col1, b.col2
FROM big_table a
JOIN small_table b ON a.id = b.id
WHERE a.created_at > '2023-01-01'
```

You might want to run the part referencing `big_table` in Doris while leaving the join to `small_table` in MySQL.

### 5.1 Identifying Big Tables

```java
SQLSelectQueryBlock queryBlock = selectStmt.getSelect().getQueryBlock();
SQLTableSource rootFrom = queryBlock.getFrom();

// A simple example visitor or manual walk to see if big_table is used
final Set<String> bigTables = new HashSet<>();
bigTables.add("big_table");

// Pseudocode: examine the 'rootFrom' and see if it matches big_table
// If it does, extract that portion of the query
```

### 5.2 Example: Removing the Join to Handle Externally

In a simplistic approach, you might:
- Extract the JOIN condition for the big table
- Turn the rest of the query into a “local” query that expects the big table’s result as a temporary table or parameter

```java
// Suppose we found that left side is big_table, and right side is small_table
SQLJoinTableSource join = (SQLJoinTableSource) rootFrom;

SQLTableSource left = join.getLeft();   // big_table
SQLTableSource right = join.getRight(); // small_table
SQLExpr joinCondition = join.getCondition();

// We'll remove the join from the original AST:
queryBlock.setFrom(right); // effectively ignoring big_table in the FROM

// Now 'left' part is what we will run in Doris:
String bigTableOnlySql = "SELECT a.* FROM big_table a WHERE a.created_at > '2023-01-01'";
System.out.println("Run this in Doris: " + bigTableOnlySql);

// Meanwhile, the new (modified) MySQL query is basically:
String localSql = SQLUtils.toSQLString(selectStmt, JdbcConstants.MYSQL);
System.out.println("Remaining local query in MySQL: " + localSql);

// Combine result sets in your Java code as needed
```

This example is simplified (in real usage, you would handle the columns carefully and do a real plan for merging the data). But it shows how you can mutate the AST.

---

## 6. Using Visitors for Deeper Analysis

Druid also supports the **Visitor** pattern, letting you traverse the tree systematically rather than manually checking node types.

### 6.1 SchemaStatVisitor

`SchemaStatVisitor` is a built-in visitor that collects which tables, columns, conditions, etc., appear in the SQL:

```java
import com.alibaba.druid.stat.TableStat;
import com.alibaba.druid.sql.visitor.SchemaStatVisitor;

SchemaStatVisitor statVisitor = SQLUtils.createSchemaStatVisitor(dbType);
statement.accept(statVisitor);

// Show all tables
System.out.println("Tables = " + statVisitor.getTables());
// Show all columns
System.out.println("Columns = " + statVisitor.getColumns());
// Show conditions
System.out.println("Conditions = " + statVisitor.getConditions());
```

For example, if the SQL is `SELECT a.col1, b.col2 FROM big_table a JOIN small_table b ...`, you might see:

```
Tables = {big_table=Select, small_table=Select}
Columns = [big_table.col1, small_table.col2]
Conditions = [big_table.id = small_table.id, big_table.created_at > 2023-01-01]
```

### 6.2 Custom Visitor

If you need more customized traversal or rewriting, you can create your own visitor:

```java
import com.alibaba.druid.sql.ast.SQLObject;
import com.alibaba.druid.sql.visitor.MySqlASTVisitorAdapter;

public class TableAliasVisitor extends MySqlASTVisitorAdapter {

    @Override
    public boolean visit(com.alibaba.druid.sql.ast.statement.SQLExprTableSource x) {
        // For each table, do something, e.g. store info in a map
        System.out.println("Found table reference: " + x.getExpr() 
            + " alias=" + x.getAlias());
        return true; // continue visit
    }

    // override other methods as needed
}
```

Then apply it:

```java
TableAliasVisitor visitor = new TableAliasVisitor();
statement.accept(visitor);
```

---

## 7. Formatting / Re-Printing SQL

You can convert the AST back to a well-formatted SQL string:

```java
String dbType = JdbcConstants.MYSQL;
String formattedSql = SQLUtils.toSQLString(statement, dbType);
System.out.println("Formatted:\n" + formattedSql);
```

You can also specify uppercase or lowercase output using `SQLUtils.DEFAULT_FORMAT_OPTION` or `SQLUtils.DEFAULT_LCASE_FORMAT_OPTION`:

```java
String lowerCase = SQLUtils.format(sql, dbType, SQLUtils.DEFAULT_LCASE_FORMAT_OPTION);
System.out.println("Lowercase:\n" + lowerCase);
```

---

## 8. Handling Subqueries

If your query has a subquery, for example:

```sql
SELECT e.name,
       e.salary,
       (SELECT o.name FROM org o WHERE o.org_id = e.org_id) AS orgName
FROM employee e
JOIN sub_table x ON x.id = e.xid
WHERE e.salary > 1000
ORDER BY e.salary
```

Inside the AST, that subquery `(SELECT o.name ...)` is typically an `SQLSelect` inside an expression (often `SQLInSubQueryExpr`, `SQLExistsExpr`, or just an inline subquery expression). To locate it, you can:

- Inspect each `SQLSelectItem`
- If its expression is an instance of `SQLSelect` or wraps a `SQLSelect`
- Recursively parse or transform that subquery

For example:

```java
for (SQLSelectItem item : queryBlock.getSelectList()) {
    SQLExpr expr = item.getExpr();
    if (expr instanceof SQLQueryExpr) {
        SQLSelect subSelect = ((SQLQueryExpr) expr).getSubSelect();
        // This is your subquery. You can do the same things:
        SQLSelectQueryBlock subQueryBlock = subSelect.getQueryBlock();
        // Possibly rewrite subQueryBlock, or route it to Doris, etc.
    }
}
```

---

## 9. Putting It All Together

In summary, **Alibaba Druid** gives you:

1. **Fast, production-ready** SQL parsing into an AST.
2. **Visitor pattern** to traverse or transform the AST.
3. **Schema analysis** (tables/columns) with `SchemaStatVisitor`.
4. **AST rewriting** so you can separate big tables from smaller ones.
5. **Re-generation** of SQL text from the AST.

A typical flow in your scenario (splitting query between Doris and MySQL) might look like this:

1. **Parse** the user’s complex SQL.
2. **Analyze** the AST to see which tables are big (and thus stored in Doris) vs small (MySQL).
3. If needed, **split** the query:
   - Extract subqueries referencing big tables to Doris.  
   - Modify or remove them from the original AST so that the remaining part can run in MySQL (or run them as correlated subqueries, etc.).
4. **Generate** the partial queries for Doris and MySQL.
5. **Execute** them in their respective databases.
6. **Combine or join** the partial results in application memory (or via a smaller bridging query if feasible).

This approach is far more robust than hand-string manipulation because you are dealing with a structured tree, can see exactly which columns/tables are involved, and can rebuild valid SQL automatically without worrying about parentheses, spacing, or subtle string mistakes.

---

### Reference Links

- **[Druid SQL Parser Wiki (GitHub)](https://github.com/alibaba/druid/wiki/SQL-Parser)**
- **[Druid SQL AST Explanation](https://github.com/alibaba/druid/wiki/Druid_SQL_AST)**
- **[SQL Schema Repository](https://github.com/alibaba/druid/wiki/SQL_Schema_Repository)** (for advanced use in column resolution)
- **[SchemaStatVisitor](https://github.com/alibaba/druid/wiki/SchemaStatVisitor)** for table/column stats
- **[SQL Formatting](https://github.com/alibaba/druid/wiki/SQL_Format)**

---

**In short**, by leveraging Alibaba Druid’s parsing and AST features, you can systematically dissect large or complex queries, route heavy parts to a faster analytic store, keep other parts in MySQL, and then merge results without resorting to manual string hacks.