## What does this code do?

1. It **creates an in-memory schema** called `rootSchema` using Calcite’s `Frameworks.createRootSchema`.

2. It **defines a simple table** (`SimpleTable`) with two columns: `ID` (INTEGER) and `NAME` (VARCHAR).

3. It **populates** that table with some hard-coded rows: `{1, "Alice"}, {2, "Bob"}, {3, "Charlie"}`.

4. It uses Calcite’s 

   Planner

    APIs to:

   - **Parse** a SQL statement into a Calcite `SqlNode`.
   - **Validate** the statement (e.g., check column references, types).
   - **Convert** the validated AST into a **Relational Algebra** form (`RelNode`).
   - **Explain** (print) the resulting execution plan.

In short, this is a **self-contained demonstration** of how to use Calcite for **SQL-to-relational** translation and plan explanation, without any underlying database.

------

## Code Walkthrough

### 1. Main Class: `CalciteSimpleExample`

```java
public class CalciteSimpleExample {
    public static class User {
        // A simple POJO with fields 'id' and 'name'
    }
    ...
}
```

- The `User` class is just a plain Java object (POJO) with `id` and `name`. It’s not strictly used in the table scanning, but it shows how you might store row data in a real scenario.

### 2. The `SimpleTable` class

```java
static class SimpleTable extends AbstractTable implements ScannableTable {
    private final RelDataType rowType;

    SimpleTable(RelDataType rowType) {
        this.rowType = rowType;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
        return rowType;
    }

    @Override
    public Enumerable<Object[]> scan(DataContext root) {
        return Linq4j.asEnumerable(new Object[][] {
            {1, "Alice"},
            {2, "Bob"},
            {3, "Charlie"}
        });
    }
}
```

- **`AbstractTable`**: A Calcite convenience class that partially implements `Table`.
- **`ScannableTable`**: Indicates that the table can provide an `Enumerable` of rows.
- `rowType`: Stores the schema (columns, types) of the table.
- `scan(...)`: Returns an `Enumerable<Object[]>` with the row data. Here, it’s statically defined as three rows of `[ID, Name]`.

### 3. The `main` method

```java
public static void main(String[] args) throws Exception {
    ...
}
```

#### a. Create the root schema

```java
SchemaPlus rootSchema = Frameworks.createRootSchema(true);
```

- `SchemaPlus` is the top-level Calcite schema object.
- `createRootSchema(true)` creates a root schema that allows modifications (i.e., you can add tables).

#### b. Create a type factory and define the table’s row types

```java
RelDataTypeFactory typeFactory = new SqlTypeFactoryImpl(RelDataTypeSystem.DEFAULT);
RelDataType intType = typeFactory.createSqlType(SqlTypeName.INTEGER);
RelDataType stringType = typeFactory.createSqlType(SqlTypeName.VARCHAR);

RelDataType rowType = typeFactory.builder()
    .add("ID", intType)
    .add("NAME", stringType)
    .build();
```

- `SqlTypeFactoryImpl` is a standard Calcite factory for SQL types.
- `SqlTypeName.INTEGER` and `SqlTypeName.VARCHAR` are standard SQL types in Calcite.
- The builder creates a row type with two fields: `ID` (integer) and `NAME` (varchar).

#### c. Add the table to the schema

```java
rootSchema.add("USERS", new SimpleTable(rowType));
```

- This is how you register a new table named `USERS` in the `rootSchema`.
- `new SimpleTable(rowType)` is the custom table that returns the row data in `scan()`.

#### d. Create a parser config and a `FrameworkConfig`

```java
SqlParser.Config parserConfig = SqlParser.config()
    .withCaseSensitive(false);

FrameworkConfig config = Frameworks.newConfigBuilder()
    .parserConfig(parserConfig)
    .defaultSchema(rootSchema)
    .build();
```

- `parserConfig` sets **case-insensitivity** for the parser.
- `FrameworkConfig` bundles together the parser config, default schema, and other Calcite settings needed by the planner.

#### e. Run a few queries

```java
String[] queries = {
    "SELECT * FROM USERS",
    "SELECT * FROM users WHERE ID = 1",
    "SELECT name FROM USERS WHERE id > 1",
    "SELECT COUNT(*) FROM users"
};

Planner planner = Frameworks.getPlanner(config);
```

- We create an array of SQL statements to run.
- **`Planner`**: The main interface for parsing, validating, and converting SQL to relational expressions.

#### f. For each query

```java
for (String sql : queries) {
    System.out.println("\nExecuting query: " + sql);
    try {
        // Parse
        SqlNode parse = planner.parse(sql);
        // Validate
        SqlNode validate = planner.validate(parse);
        
        // Convert to RelNode
        RelNode rel = planner.rel(validate).project();
        
        System.out.println("Execution plan:");
        rel.explain(new RelWriterImpl(new PrintWriter(System.out, true)));
    } catch (Exception e) {
        System.out.println("Error executing query: " + e.getMessage());
    }
    
    // Reset the planner for next query
    planner.close();
    planner = Frameworks.getPlanner(config);
}
```

- `parse(...)` -> `SqlNode`.
- `validate(...)` -> ensures columns and types are correct.
- `planner.rel(...)` -> `RelRoot`. `.project()` retrieves the top-level `RelNode`.
- `rel.explain(...)` prints the relational plan.
- After each query, the planner is closed and re-initialized.

------

## Imports in the Calcite-Only Example

1. `org.apache.calcite.schema.*`
   - Provides interfaces for schemas and tables (e.g., `Schema`, `Table`, `ScannableTable`).
2. `org.apache.calcite.schema.impl.AbstractTable`
   - A base class that partially implements `Table`.
3. `org.apache.calcite.linq4j.Enumerable` / `org.apache.calcite.linq4j.Linq4j`
   - Calcite’s extension of LINQ-like operations, returning row sets as `Enumerable`.
4. `org.apache.calcite.rel.RelNode`
   - The fundamental representation of a relational expression (logical or physical plan).
5. `org.apache.calcite.tools.*`
   - Provides classes like `Frameworks`, `Planner`, `FrameworkConfig` for building the planning environment.
6. `org.apache.calcite.rel.type.RelDataType`, `RelDataTypeFactory`, `RelDataTypeSystem`
   - Represent Calcite’s type system for relational data.
7. `org.apache.calcite.sql.SqlNode`
   - A node in Calcite’s SQL AST (after parsing).
8. `org.apache.calcite.sql.parser.SqlParseException`, `SqlParser`
   - The SQL parser and exceptions for syntax issues.
9. `org.apache.calcite.sql.type.BasicSqlType`, `SqlTypeFactoryImpl`, `SqlTypeName`
   - Implementations for SQL types such as VARCHAR, INTEGER, etc.
10. **`org.apache.calcite.DataContext`**

- A context for execution, holding runtime parameters. Used in `scan(...)`.

1. **`org.apache.calcite.rel.externalize.RelWriterImpl`**

- Used to print (explain) the `RelNode` plan.

1. **`java.io.PrintWriter`**

- Standard Java class for output text.

1. **`java.util.\*`**

- Collections like `List`, `ArrayList`, etc.