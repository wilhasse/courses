Yes, **Alibaba Druid** can parse queries with subqueries, including subqueries in the `WHERE` clause (e.g., `IN (SELECT ...)`), in the `FROM` clause (subselect as a table source), and others. Below are a few **examples** illustrating how to parse and inspect subqueries using the **MySQL** dialect parser in Druid.

------

## 1. Subquery in `WHERE` Clause

### Example SQL

```sql
SELECT emp_id, first_name, salary
FROM employees
WHERE salary > (
  SELECT AVG(salary)
  FROM employees
  WHERE department_id = 50
)
```

### Parsing & Accessing the AST

```java
import com.alibaba.druid.sql.ast.SQLExpr;
import com.alibaba.druid.sql.ast.SQLStatement;
import com.alibaba.druid.sql.ast.expr.SQLBinaryOpExpr;
import com.alibaba.druid.sql.ast.expr.SQLQueryExpr;
import com.alibaba.druid.sql.ast.statement.SQLSelectStatement;
import com.alibaba.druid.sql.ast.statement.SQLSelectQueryBlock;
import com.alibaba.druid.sql.dialect.mysql.parser.MySqlStatementParser;

...

String sql = "SELECT emp_id, first_name, salary\n" +
             "FROM employees\n" +
             "WHERE salary > (\n" +
             "  SELECT AVG(salary)\n" +
             "  FROM employees\n" +
             "  WHERE department_id = 50\n" +
             ")";

MySqlStatementParser parser = new MySqlStatementParser(sql);
List<SQLStatement> stmtList = parser.parseStatementList();

// Typically we have 1 statement here
SQLStatement stmt = stmtList.get(0);
SQLSelectStatement selectStmt = (SQLSelectStatement) stmt;

// The main query block
SQLSelectQueryBlock mainQueryBlock = (SQLSelectQueryBlock) selectStmt.getSelect().getQuery();

// The WHERE clause
SQLExpr whereExpr = mainQueryBlock.getWhere();
// e.g., salary > (SELECT AVG(salary) FROM employees WHERE department_id = 50)

// In Druid, this is usually a SQLBinaryOpExpr with operator ">"
if (whereExpr instanceof SQLBinaryOpExpr) {
    SQLBinaryOpExpr boExpr = (SQLBinaryOpExpr) whereExpr;
    // boExpr.getLeft() => "salary"
    // boExpr.getRight() => the subquery expression

    SQLExpr rightSide = boExpr.getRight();
    if (rightSide instanceof SQLQueryExpr) {
        SQLQueryExpr subQueryExpr = (SQLQueryExpr) rightSide;
        // subQueryExpr.getSubQuery() => SQLSelect
        SQLSelectQueryBlock subQueryBlock = (SQLSelectQueryBlock) subQueryExpr.getSubQuery().getQuery();
        // subQueryBlock corresponds to:
        //     SELECT AVG(salary) 
        //     FROM employees
        //     WHERE department_id = 50

        // For example, we can access WHERE in the subquery
        SQLExpr subWhere = subQueryBlock.getWhere();
        // subWhere => department_id = 50
    }
}
```

In the code above, Druid automatically constructs:

- A **top-level** `SQLSelectQueryBlock` for the outer query.
- A **`SQLQueryExpr`** for the subquery on the right side of `salary > (...)`.

Once you have that subquery as a `SQLQueryExpr`, you can inspect or modify it as needed.

------

## 2. Subquery in `WHERE ... IN (SELECT ...)`

### Example SQL

```sql
SELECT name
FROM student
WHERE age IN (
  SELECT age
  FROM student_archive
  WHERE is_graduated = 1
);
```

### Parsing & Inspecting

```java
String sql = "SELECT name FROM student " +
             "WHERE age IN (" +
             "  SELECT age FROM student_archive WHERE is_graduated = 1" +
             ")";

MySqlStatementParser parser = new MySqlStatementParser(sql);
List<SQLStatement> stmtList = parser.parseStatementList();
SQLSelectStatement selectStmt = (SQLSelectStatement) stmtList.get(0);

SQLSelectQueryBlock mainQueryBlock = (SQLSelectQueryBlock) selectStmt.getSelect().getQuery();

// WHERE clause might be a SQLInSubQueryExpr
SQLExpr whereExpr = mainQueryBlock.getWhere();
System.out.println(whereExpr.getClass().getName());
// often: com.alibaba.druid.sql.ast.expr.SQLInSubQueryExpr

// If we cast it:
if (whereExpr instanceof com.alibaba.druid.sql.ast.expr.SQLInSubQueryExpr) {
    com.alibaba.druid.sql.ast.expr.SQLInSubQueryExpr inExpr =
        (com.alibaba.druid.sql.ast.expr.SQLInSubQueryExpr) whereExpr;

    // The left side: "age"
    SQLExpr left = inExpr.getExpr(); 
    // The subselect
    SQLSelectQueryBlock subQueryBlock =
        (SQLSelectQueryBlock) inExpr.getSubQuery().getQuery();

    System.out.println("Main Query Column: " + left.toString());
    System.out.println("SubQuery FROM: " + subQueryBlock.getFrom());
    System.out.println("SubQuery WHERE: " + subQueryBlock.getWhere());
}
```

Here, the parser recognizes the `IN (SELECT ...)` pattern and represents it via `SQLInSubQueryExpr`. The subquery is accessible through `inExpr.getSubQuery().getQuery()`.

------

## 3. Subquery in the `FROM` Clause (Derived Table)

### Example SQL

```sql
SELECT t.dept_no, t.avg_sal
FROM (
    SELECT dept_no, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY dept_no
) AS t
WHERE t.avg_sal > 5000;
```

### Parsing & Accessing the Derived Table

```java
String sql = "SELECT t.dept_no, t.avg_sal\n" +
             "FROM (\n" +
             "  SELECT dept_no, AVG(salary) AS avg_sal\n" +
             "  FROM employees\n" +
             "  GROUP BY dept_no\n" +
             ") AS t\n" +
             "WHERE t.avg_sal > 5000";

MySqlStatementParser parser = new MySqlStatementParser(sql);
List<SQLStatement> stmts = parser.parseStatementList();
SQLSelectStatement selectStmt = (SQLSelectStatement) stmts.get(0);

SQLSelectQueryBlock mainBlock = (SQLSelectQueryBlock) selectStmt.getSelect().getQuery();

// The FROM is a SQLSubqueryTableSource
SQLTableSource from = mainBlock.getFrom();
System.out.println(from.getClass().getName());
// com.alibaba.druid.sql.ast.statement.SQLSubqueryTableSource

// Cast to SQLSubqueryTableSource
SQLSubqueryTableSource subqueryTable = (SQLSubqueryTableSource) from;

// The subquery: SELECT dept_no, AVG(salary) AS avg_sal FROM employees GROUP BY dept_no
SQLSelect subSelect = subqueryTable.getSelect();
SQLSelectQueryBlock subBlock = (SQLSelectQueryBlock) subSelect.getQuery();
System.out.println("Subquery: " + subBlock.toString());

// Now we can see the columns, the GROUP BY, etc.
System.out.println("Subquery SELECT List: " + subBlock.getSelectList());
System.out.println("Subquery GROUP BY: " + subBlock.getGroupBy());
```

Druid represents this “derived table” in `FROM (SELECT ...) AS t` as a **`SQLSubqueryTableSource`**. You can then get the underlying `SQLSelectQueryBlock` to see the subquery’s `SELECT` list, `FROM`, `WHERE`, etc.

------

## 4. Modifying Subqueries

One of the main strengths of the Druid parser is that you can **modify** the AST. For example, you might want to:

1. Append an extra condition to the subquery’s `WHERE` clause.
2. Add or remove columns from the subquery’s select list.
3. Insert hints into the subquery (e.g., `/*+ someHint */`).

Here’s a small snippet showing how you could **add** a condition to the subquery’s `WHERE`:

```java
if (subBlock.getWhere() == null) {
    // subBlock.setWhere(new SQLIdentifierExpr("colX = 123"));
} else {
    // Suppose the existing WHERE is colY = 10
    // We want to add an AND colX = 123
    SQLBinaryOpExpr newWhere = new SQLBinaryOpExpr();
    newWhere.setOperator(SQLBinaryOperator.BooleanAnd);

    newWhere.setLeft(subBlock.getWhere());
    newWhere.setRight(new SQLBinaryOpExpr(
        new SQLIdentifierExpr("colX"),
        SQLBinaryOperator.Equality,
        new SQLIntegerExpr(123)
    ));
    subBlock.setWhere(newWhere);
}
```

After that, you can do:

```java
// Convert the AST back to string
String newSql = SQLUtils.toMySqlString(selectStmt);
System.out.println("Rewritten SQL: " + newSql);
```

------

## 5. Summary & Tips

1. **Yes, Alibaba Druid handles subqueries** quite well in many typical positions (WHERE, FROM, SELECT-list expressions, etc.).

2. When dealing with subqueries, you’ll commonly see AST nodes like:

   - `SQLQueryExpr` for subqueries in expressions (`> (SELECT ...)`, `IN (SELECT ...)`).
   - `SQLSubqueryTableSource` for subqueries in the `FROM` clause.
   - `SQLInSubQueryExpr` for `IN (SELECT ...)`.
   - `SQLExistsExpr` for `EXISTS (SELECT ...)`.

3. Traversal or rewriting

    can be done by:

   - Directly calling `getWhere()`, `getFrom()`, `getSelectList()`, etc.
   - Using the **visitor pattern** (`SQLASTVisitor`) for deeper queries with multiple subqueries or more complex manipulations.

4. **Take care** when rewriting subqueries: it’s often best to manipulate the AST nodes rather than do naive string concatenation. This way, your rewriting is robust to spacing, parentheses, or complex logic.

**That’s it!** With these examples, you can parse and manipulate subqueries using Alibaba Druid’s MySQL parser. If you have more advanced needs (like correlated subqueries, window functions, etc.), Druid still generally handles them—but you’ll just see different AST node types (e.g., `SQLSelectQueryBlock` with correlated references, or `SQLOver` for windowing).