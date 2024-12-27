# Source

https://calcite.apache.org/docs/tutorial.html  


Below is a simplified, **beginner-friendly** explanation of Calcite’s CSV adapter tutorial. The goal is to make it understandable even if you’ve never used Calcite before.

------

# What is Calcite?

**Apache Calcite** is a framework that provides:

- **SQL parsing** (it can understand SQL statements),
- **query optimization** (it tries to figure out the most efficient way to run your query),
- **JDBC driver** (so you can connect to Calcite as if it were a database).

**But** Calcite itself does **not** store any data. Instead, it relies on **adapters** to connect to data wherever it lives (for example, in CSV files, in a real database like MySQL, or in a custom data store). Calcite then lets you run SQL queries on that data as if it were in a single database.

------

# 1. Tutorial Overview

This tutorial focuses on **the CSV adapter**:

- It makes a **directory of CSV files** look like database tables.
- You can then run SQL queries on those CSV files via Calcite.

**Why CSV?** CSV files are easy to understand, and the CSV adapter is short enough that you can see how a Calcite adapter works. You can later use the same ideas to build adapters for other data sources.

------

# 2. Download and Build

1. **Install Java** (8 or higher) and Git.

2. Clone the Calcite repository

    from GitHub:

   ```bash
   git clone https://github.com/apache/calcite.git
   cd calcite/example/csv
   ```

3. Start the SQL shell

    that comes with Calcite (called sqlline):

   ```bash
   ./sqlline
   ```

   On Windows, you would use 

   ```
   sqlline.bat
   ```

    instead of 

   ```
   ./sqlline
   ```

   .

That’s it! You have the Calcite code and a minimal shell to run SQL queries.

------

# 3. First Queries with the CSV Adapter

Inside the `example/csv` folder, there’s a ready-to-go **model file** called `model.json`. A “model file” is just a JSON file that tells Calcite:

- *Where* your CSV files are located,
- *Which* schema factory to use,
- *What* tables or views to create (optionally).

To connect:

```sql
!connect jdbc:calcite:model=src/test/resources/model.json admin admin
```

1. `jdbc:calcite:` tells sqlline to use Calcite’s driver.
2. `model=src/test/resources/model.json` points Calcite to the JSON file that describes the schema.

### 3.1 List Available Tables

```sql
sqlline> !tables
```

You’ll see something like:

| TABLE_CAT | TABLE_SCHEM | TABLE_NAME | TABLE_TYPE   | ...  |
| --------- | ----------- | ---------- | ------------ | ---- |
|           | SALES       | DEPTS      | TABLE        | ...  |
|           | SALES       | EMPS       | TABLE        | ...  |
|           | SALES       | SDEPTS     | TABLE        | ...  |
|           | metadata    | COLUMNS    | SYSTEM TABLE | ...  |
|           | metadata    | TABLES     | SYSTEM TABLE | ...  |

There are:

- **EMPS**, **DEPTS**, **SDEPTS** tables in a schema called **SALES**
- Two system tables, **COLUMNS** and **TABLES**, for Calcite’s internal metadata.

### 3.2 Run a Simple Query

```sql
sqlline> SELECT * FROM emps;
```

You’ll see rows that come from the `EMPS.csv.gz` file. Calcite is reading that file behind the scenes.

### 3.3 JOIN and GROUP BY

```sql
sqlline> SELECT d.name, COUNT(*)
       > FROM emps AS e
       > JOIN depts AS d ON e.deptno = d.deptno
       > GROUP BY d.name;
```

Calcite will do a join between the CSV file `EMPS.csv.gz` and `DEPTS.csv`, then group by the department name.

### 3.4 Test Expressions with VALUES

```sql
sqlline> VALUES CHAR_LENGTH('Hello, ' || 'world!');
```

You get:

```
+---------+
| EXPR$0  |
+---------+
| 13      |
+---------+
```

So yes, Calcite supports string functions, numeric functions, and many other SQL features.

------

# 4. How Does Calcite Find the CSV Files?

Calcite knows nothing about CSV files by default. Instead, the **model.json** file tells Calcite:

```json
{
  "version": "1.0",
  "defaultSchema": "SALES",
  "schemas": [
    {
      "name": "SALES",
      "type": "custom",
      "factory": "org.apache.calcite.adapter.csv.CsvSchemaFactory",
      "operand": {
        "directory": "sales"
      }
    }
  ]
}
```

- **"factory"**: the Java class that creates a special “CSV schema.”
- **"operand"**: additional configuration (like `"directory": "sales"` means “look in the `sales` directory for CSV files”).

When you connect, Calcite calls `CsvSchemaFactory.create(...)`, which returns a `CsvSchema` object that scans the directory for `.csv` or `.csv.gz` files and automatically creates a table for each file.

------

# 5. Defining Tables and Views in the Model

### 5.1 Automatic Tables

As mentioned, the CSV schema automatically creates a table for each CSV file in the directory. That’s why you see EMPS, DEPTS, and SDEPTS without having to list them in the JSON.

### 5.2 Custom Views

You can also define a **view** (similar to a saved query in a traditional database) by adding a block like:

```json
"tables": [
  {
    "name": "FEMALE_EMPS",
    "type": "view",
    "sql": "SELECT * FROM emps WHERE gender = 'F'"
  }
]
```

Now you can query:

```sql
SELECT e.name, d.name
FROM FEMALE_EMPS AS e
JOIN DEPTS AS d ON e.deptno = d.deptno;
```

It’s like a virtual table that filters the real table.

### 5.3 Custom Tables

If you don’t want Calcite to automatically scan for `.csv` files, you can define a **custom table** in JSON:

```json
{
  "name": "EMPS",
  "type": "custom",
  "factory": "org.apache.calcite.adapter.csv.CsvTableFactory",
  "operand": {
    "file": "sales/EMPS.csv.gz",
    "flavor": "scannable"
  }
}
```

Here, you manually specify the CSV file and how you want to read it. This can be useful if you have custom parameters or advanced logic.

------

# 6. Optimizing Queries: Planner Rules

If your CSV files are huge (millions of rows), you probably want to skip reading unneeded columns or rows. Calcite can push those decisions (e.g., “only read column X”) to the CSV adapter if you add **planner rules**.

### 6.1 What Are Planner Rules?

- Calcite’s query planner looks at your SQL query and tries to transform it into an efficient plan.
- A **rule** is basically “if you see pattern A in the query, replace it with pattern B.”
- For CSV, there’s a rule that, for example, recognizes “SELECT only a few columns” and avoids reading all columns.

### 6.2 Example: Project Pushdown

If the schema is created with `"flavor": "translatable"`, you get a specialized table object (`CsvTranslatableTable`) that supports rules like `CsvProjectTableScanRule`. This rule looks for a “project” (i.e., “select only certain columns”) and modifies the plan so that only the needed columns are read.

That’s why `EXPLAIN PLAN FOR SELECT name FROM emps;` might show a different result when using `flavor = "translatable"`. It indicates that Calcite is pushing down the column selection into the CSV scan, which is more efficient.

------

# 7. The Query Optimization Process

Calcite’s planner:

- Tries many possible transformations (rules) in no fixed order.
- Uses a **cost model** to figure out which transformation is the cheapest plan.
- Allows you to mix and match many rule sets for different data sources.

So if you combine the CSV adapter with, say, a MySQL adapter or a custom adapter, Calcite can use all the rules together to find the best plan across multiple sources.

------

# 8. JDBC Adapter

Calcite also provides a **JDBC adapter** that can talk to a real relational database. For example:

```json
{
  "version": "1.0",
  "defaultSchema": "FOODMART",
  "schemas": [
    {
      "name": "FOODMART",
      "type": "custom",
      "factory": "org.apache.calcite.adapter.jdbc.JdbcSchema$Factory",
      "operand": {
        "jdbcDriver": "com.mysql.jdbc.Driver",
        "jdbcUrl": "jdbc:mysql://localhost/foodmart",
        "jdbcUser": "foodmart",
        "jdbcPassword": "foodmart"
      }
    }
  ]
}
```

Now, a table in the MySQL `foodmart` database becomes a table in Calcite’s “FOODMART” schema. Calcite tries to **push down** as much processing (filters, joins, aggregates) to MySQL.

------

# 9. The “Cloning” JDBC Adapter

The **cloning** adapter is a hybrid approach:

- It reads data once from a JDBC source (for example, MySQL),
- Caches or clones that data in an in-memory table managed by Calcite,
- Then queries go against that in-memory table.

For example:

```json
{
  "version": "1.0",
  "defaultSchema": "FOODMART_CLONE",
  "schemas": [
    {
      "name": "FOODMART_CLONE",
      "type": "custom",
      "factory": "org.apache.calcite.adapter.clone.CloneSchema$Factory",
      "operand": {
        "jdbcDriver": "com.mysql.jdbc.Driver",
        "jdbcUrl": "jdbc:mysql://localhost/foodmart",
        "jdbcUser": "foodmart",
        "jdbcPassword": "foodmart"
      }
    }
  ]
}
```

Alternatively, you can “clone” any other existing Calcite schema (not just JDBC). This might be used for caching, offline analysis, or performance experiments.

------

# 10. Further Topics

- The **Calcite adapter specification** is the official guide to building custom adapters for different file formats or data stores.
- You can embed Calcite in your application and leverage:
  - Custom table definitions,
  - Custom rules for optimization,
  - Advanced cost models,
  - Materialized views (precomputed query results), and more.

------

## Final Thoughts

1. **Calcite** is like a “brain” that knows how to parse, plan, and optimize queries.

2. **Adapters** tell Calcite how to read data from different places (CSV files, JDBC databases, NoSQL, etc.).

3. **Model files** (JSON) configure schemas—where to find data and how to represent it as tables/views.

4. The 

   CSV adapter

    is a simple example but demonstrates the entire flow:

   - A schema factory reads a directory for `.csv` files,
   - Each file becomes a table,
   - You can define **views** in the JSON,
   - Planner rules can optimize queries further.

This tutorial shows that once Calcite knows how to talk to your data (via an adapter), you get a **full SQL interface** over that data with relatively little code. It’s a powerful way to unify data access and query optimization across many different data systems. Enjoy exploring!
