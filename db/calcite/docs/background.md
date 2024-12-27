# Source

https://calcite.apache.org/docs/

Below is a step-by-step, detailed explanation of the Apache Calcite documentation excerpt you’ve provided. The goal is to clarify both the concepts and how they fit together within Calcite.

------

## 1. What is Apache Calcite?

### 1.1 A Dynamic Data Management Framework

- Apache Calcite is described as a “dynamic data management framework.”

- It contains many of the typical components you would find in a database management system (DBMS)—such as a SQL parser, a query validator, an optimizer, a JDBC driver—

  but

   it deliberately omits:

  1. **Data storage**: Calcite does not store the data itself.
  2. **Data-processing algorithms**: Calcite typically delegates the actual data processing to external engines or libraries.
  3. **Metadata repository**: Calcite does not maintain its own global metadata storage.

This intentional separation (Calcite not storing or processing data) allows it to serve as a **mediation layer**: it can sit in front of one or more data sources (e.g., a relational database, a CSV file, or even a custom data source) and unify queries across all of them.

------

## 2. Why This Separation is Useful

### 2.1 Mediation Between Different Data Sources

Calcite can connect to numerous data sources—traditional databases (e.g., MySQL, PostgreSQL), file-based sources (CSV, JSON), in-memory Java data structures, NoSQL engines, etc. In each case, Calcite can present a unified SQL interface to the end user or application.

### 2.2 Foundation for Building a Database

Because Calcite provides all the SQL and optimization tooling, if you ever want a full-blown database system, you can “just add data” and a storage layer. Calcite can handle the parsing, validation, and planning aspects for you.

------

## 3. Basic Example: In-Memory “Database”

Below is the example from the documentation (condensed and then explained in detail):

```java
public static class HrSchema {
  public final Employee[] emps = {};
  public final Department[] depts = {};
}

Class.forName("org.apache.calcite.jdbc.Driver");
Properties info = new Properties();
info.setProperty("lex", "JAVA");
Connection connection =
    DriverManager.getConnection("jdbc:calcite:", info);

CalciteConnection calciteConnection =
    connection.unwrap(CalciteConnection.class);

SchemaPlus rootSchema = calciteConnection.getRootSchema();

// Register a reflective schema using the Java class HrSchema
Schema schema = new ReflectiveSchema(new HrSchema());
rootSchema.add("hr", schema);

Statement statement = calciteConnection.createStatement();

// Example SQL query
ResultSet resultSet = statement.executeQuery(
    "select d.deptno, min(e.empid)\n"
  + "from hr.emps as e\n"
  + "join hr.depts as d\n"
  + "  on e.deptno = d.deptno\n"
  + "group by d.deptno\n"
  + "having count(*) > 1"
);

print(resultSet);
resultSet.close();
statement.close();
connection.close();
```

### 3.1 What’s Happening Here

1. **Create the Schema**:

   ```java
   public static class HrSchema {
     public final Employee[] emps = {};
     public final Department[] depts = {};
   }
   ```

   - A simple Java class `HrSchema` has arrays of `Employee` and `Department`. Each array becomes, in effect, a “table.”

2. **Load the JDBC Driver**:

   ```java
   Class.forName("org.apache.calcite.jdbc.Driver");
   ```

   - This is the Calcite-specific JDBC driver.

3. **Create a Connection**:

   ```java
   Connection connection = DriverManager.getConnection("jdbc:calcite:", info);
   ```

   - The URL `"jdbc:calcite:"` tells Java’s `DriverManager` to use Calcite’s driver.
   - The `Properties` object can hold Calcite-specific options (e.g., SQL dialect conventions).

4. **Unwrap CalciteConnection**:

   ```java
   CalciteConnection calciteConnection = connection.unwrap(CalciteConnection.class);
   ```

   - A `CalciteConnection` gives you access to Calcite-specific methods, such as obtaining the root schema.

5. **Register the Schema**:

   ```java
   Schema schema = new ReflectiveSchema(new HrSchema());
   rootSchema.add("hr", schema);
   ```

   - This is the crucial step: it tells Calcite that the `HrSchema` Java object is a **schema**, and that each array field (`emps`, `depts`) is a **table**.
   - “Reflective schema” means Calcite automatically detects public fields and methods that behave like tables.

6. **Issue a SQL Query**:

   ```java
   Statement statement = calciteConnection.createStatement();
   ResultSet resultSet = statement.executeQuery(...);
   ```

   - We now run SQL that references `hr.emps` and `hr.depts`.
   - Notice that `hr` is the name we gave to the schema, and `emps`/`depts` are the “tables” derived from the Java arrays.

### 3.2 No Actual Database

- There is **no** real database behind the scenes. It’s all in Java memory (the arrays `emps` and `depts`).
- Calcite uses its built-in in-memory processing capabilities (powered by [Linq4j](https://calcite.apache.org/lincq4j)) to perform the SQL logic (JOIN, GROUP BY, etc.).

------

## 4. Replacing In-Memory with JDBC

Next, the documentation shows that instead of using a reflective schema (in-memory), you can point Calcite at a **real relational database** (e.g., MySQL):

```java
Class.forName("com.mysql.jdbc.Driver");
BasicDataSource dataSource = new BasicDataSource();
dataSource.setUrl("jdbc:mysql://localhost");
dataSource.setUsername("username");
dataSource.setPassword("password");

Schema schema = JdbcSchema.create(rootSchema, "hr", dataSource, null, "name");
```

- **What changed**:
  - We load a different JDBC driver (MySQL).
  - We create a `DataSource` that points to a real MySQL database.
  - We then create a `JdbcSchema` which makes that MySQL database appear as the “hr” schema in Calcite.
  - **The exact same SQL query** can now run, but the data will come from MySQL tables, not local arrays.
- **Behind the scenes**: Calcite uses its query optimization engine to **push** operations—like `JOIN` and `GROUP BY`—down into MySQL whenever possible. This offloads as much work as it can to the underlying database, rather than doing it in Java.

------

## 5. Extending Calcite with Adapters

### 5.1 What is an Adapter?

- An adapter is how Calcite knows how to interpret a particular data source. For example:
  - The **ReflectiveSchema adapter** interprets Java objects/fields as tables.
  - The **JDBC adapter** interprets a relational database via standard SQL access.
  - The **CSV adapter** (found in Calcite’s `example/csv` module) interprets CSV files as tables.

### 5.2 Writing Your Own Adapter

- If you have a custom data store—say, a proprietary format or a specialized system—you can write an adapter.
- The adapter instructs Calcite on:
  - How to list what “tables” are available in the data source.
  - How to read data from those tables.
  - (Optionally) which SQL operations (filter, project, aggregate, join, etc.) can be **pushed down** for more efficient execution.

------

## 6. Optimizer Rules and Custom Operators

### 6.1 Optimizer Rules

- Calcite uses a **cost-based** optimization engine (based on the [Volcano planner](https://calcite.apache.org/docs/planner.html)).
- You can add your own rules to teach Calcite:
  1. How to rewrite logical plans into more optimal forms for your data source.
  2. How to substitute your own physical operators if your data source or system has specialized capabilities.

### 6.2 Custom Operators

- You can register new functions, aggregations, or even table functions that Calcite can handle.
- For instance, if your system has a custom join algorithm or a specialized indexing strategy, you can add them via new rules and operators.

------

## 7. CSV Adapter and Tutorials

### 7.1 CSV Adapter as a Template

- Calcite ships with a CSV adapter in the 

  ```
  example/csv
  ```

   directory, which:

  - Allows reading CSV files as if they were database tables.
  - Is simple enough to serve as a reference if you want to build your own adapter.

### 7.2 Tutorials and HOWTOs

- The Calcite website includes a [tutorial](https://calcite.apache.org/docs/tutorial.html) and a [HOWTO](https://calcite.apache.org/docs/howto.html) for more detailed guidance.
- These resources show how to configure adapters, set up JSON model files, and embed Calcite into your application.

------

## 8. Status and Features

The doc concludes by listing some of Calcite’s **features** that are ready to use:

1. **Query parser, validator, and optimizer** (the core of Calcite).

2. **Support for reading models in JSON** (you can define your data sources in JSON and load them at runtime).

3. **Many standard functions and aggregates** (e.g., `COUNT`, `SUM`, `AVG`, etc.).

4. **JDBC queries against Linq4j and JDBC back-ends** (makes it very flexible for bridging data sources).

5. **Linq4j front-end** (for those who prefer a .NET/Java-like LINQ approach).

6. SQL features

   :

   - Basic SELECT-FROM-WHERE,
   - JOIN syntax,
   - GROUP BY including complex grouping sets,
   - DISTINCT aggregates,
   - FILTER on aggregates,
   - HAVING,
   - ORDER BY with NULLS FIRST/LAST,
   - Set operations (UNION, INTERSECT, MINUS),
   - Sub-queries (including correlated sub-queries),
   - Windowed aggregates,
   - LIMIT (syntax similar to Postgres).

7. **Local and remote JDBC drivers** via [Avatica](https://calcite.apache.org/avatica/).

8. **Several adapters** (e.g., CSV, MongoDB, Cassandra, Elasticsearch, etc.).

------

## 9. Takeaways

1. **Calcite is Lightweight**: It’s all about query parsing, validation, and optimization—not data storage.
2. **Adapter-Driven**: You plug in adapters for the data sources you want (JDBC databases, CSV files, NoSQL, etc.).
3. **Easy to Embed**: You can embed Calcite in any JVM-based environment to offer an SQL interface to data—regardless of how or where that data is stored.
4. **Extensible**: By adding custom rules and adapters, you can tailor Calcite to almost any kind of data or specialized optimization technique.

------

### In Summary

- **Apache Calcite** acts as a flexible SQL engine and optimizer, mediating between applications and virtually any data source.
- In the simplest usage, you can treat **Java arrays** as tables with Calcite’s reflective schema.
- To connect to a **real database**, just switch to the **JDBC adapter**, and Calcite can push down queries to that database.
- To handle other storage formats (CSV, JSON, NoSQL, custom data stores), you either use or build an **adapter** so Calcite knows how to interpret and optimize queries against that data.
- Finally, you can extend Calcite with **optimizer rules** to achieve custom functionality and improved performance.

This is what the documentation snippet conveys and why Calcite is often used as the backbone for SQL-based federation, integration, or custom database engines.