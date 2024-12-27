# Source

https://calcite.apache.org/docs/adapter.html  

Below is a **plain-language** explanation of Calcite’s **adapters**—what they are, why they matter, and how they fit into the bigger picture of Calcite.

------

## 1. What is an Adapter in Calcite?

**Short answer:** An **adapter** is a piece of code that teaches Calcite how to talk to a particular kind of data source.

When you use Calcite to run SQL queries, Calcite itself doesn’t store or process data—it needs to connect to **something** that has data (like CSV files, Cassandra, MySQL, etc.). The adapter is that “something” that **bridges** Calcite with the external data format or system.

1. **Calcite side**: Expects to see “tables” and “columns” that it can run SQL on.
2. **Data-source side**: Might be CSV files on disk, a NoSQL database, a streaming engine, or anything else.
3. **Adapter**: Translates back and forth—Calcite issues queries in a “relational algebra” style, and the adapter performs the necessary reads (and possibly writes) in the target data source.

------

## 2. Schema Adapters vs. Drivers (and other terms)

### 2.1 Schema Adapters

A **schema adapter** is a special type of adapter that presents data as a **schema** containing **tables**. For example:

- **CSV adapter**: Looks in a directory for CSV files; each file is treated like a table.
- **MongoDB adapter**: Connects to a MongoDB database; each collection is a table.
- **JDBC adapter**: Connects to an existing relational database (like MySQL or Postgres) via JDBC; each DB table is a Calcite table.

In the Calcite docs, you might see references to “schema adapters.” This is simply an adapter that “exposes” data as if it were a normal SQL schema with tables.

### 2.2 JDBC Driver vs. Adapter

- A 

  JDBC driver

   is how you (as a 

  client

  ) connect 

  to Calcite

   itself: you do something like

  ```
  Connection conn = DriverManager.getConnection("jdbc:calcite:model=...");
  ```

  to run queries against Calcite.

- An **adapter** is how **Calcite** connects “downwards” to your data. For instance, if you want to read from a Cassandra database, you’d use the **Cassandra adapter**.

So, if you imagine the flow of data:

```
[Your Application] --(JDBC driver)--> [Calcite] --(Adapter)--> [Your Data: CSV, Cassandra, etc.]
```

------

## 3. Examples of Adapters

Calcite has many **built-in** or **ready-to-use** adapters:

1. **CSV adapter**
   - Found under `example/csv` in Calcite’s GitHub repo.
   - Makes CSV files appear as tables.
2. **Cassandra adapter**
   - Connects to Apache Cassandra.
   - Allows Calcite to issue SQL queries that get translated into Cassandra queries where possible.
3. **MongoDB adapter**
   - Treats MongoDB collections as tables.
   - Calcite can push filters/aggregations into Mongo for efficiency.
4. **JDBC adapter**
   - Connects to an existing relational database (MySQL, Postgres, Oracle, etc.).
   - Many queries can be “pushed down” directly to that DB.
5. **Elasticsearch adapter**
   - Allows you to run SQL queries on an Elasticsearch index.

…and several more (Redis, Solr, Spark, etc.). Each adapter is specialized for that data source.

------

## 4. How Adapters Work Internally

1. **Schema Factory**
    Each adapter typically has a **factory class** (e.g., `CassandraSchemaFactory`) that tells Calcite:
   - Which tables exist (or how to discover them).
   - How to fetch data from those tables.
2. **Table Classes**
    Once Calcite knows “there is a table named `xyz`,” it might create a **Table object** that knows how to read rows from `xyz`. For instance, a CSV Table will read lines from a `.csv` file and parse them into rows.
3. **Pushing Down Operations**
    If the data source can do filtering or grouping more efficiently, the adapter might have “rules” telling Calcite how to push those operations down. For example, the MongoDB adapter tries to push `$match` or `$group` operations into Mongo so you only retrieve the data you need.
4. **Relational Algebra**
    Under the hood, Calcite translates your SQL into a tree of relational operations (scan, filter, project, join, etc.). Adapters can implement special versions of these operations to run them in the underlying data source.

------

## 5. How to Use an Adapter

### 5.1 JSON Model File

Often, you configure an adapter in a **JSON model file**, something like:

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

- **factory**: Points to the adapter’s Java class (like `CsvSchemaFactory`).
- **operand**: Settings for that adapter (like `"directory": "sales"` for CSV files).

### 5.2 Connect via JDBC

You can then connect from your app using:

```java
Connection conn = DriverManager.getConnection(
  "jdbc:calcite:model=/path/to/model.json");
```

Now, if the model references a CSV adapter, you can run:

```sql
SELECT * FROM sales.emps;
```

…and Calcite will read from `sales/EMPS.csv`.

------

## 6. Making Your Own Adapter

If your data is **not** in a standard place (maybe an in-memory data structure or a custom file format), you can **write your own adapter**. The main steps are:

1. **Write a SchemaFactory**
   - Tells Calcite “how to build a schema” from your data source.
   - e.g., parse a config parameter pointing to a location, create a `MyCustomSchema`.
2. **Implement Table** (or sub-interfaces like **FilterableTable**, **ProjectableFilterableTable**, or **TranslatableTable**)
   - For simple use cases, your adapter can just read **all** rows.
   - For advanced use cases, you let Calcite push filters/aggregations so you only read the needed data.
3. **(Optional) Add Planner Rules**
   - If your data source can handle certain operations (like advanced filtering or indexing) efficiently, you can write planner rules that rewrite queries to use them.

------

## 7. Why Adapters Are Useful

- **Unify Data Access**: You can run a single SQL query that **joins** data from a CSV file, a Mongo collection, and a MySQL table, all in one shot.
- **Optimization**: Calcite tries to push down heavy lifting (filters, joins, grouping) to the underlying system if the adapter can handle it.
- **Extensibility**: If your environment or data format is unique, you can build a custom adapter. Calcite takes care of the rest (parsing SQL, optimizing, etc.).

------

## 8. Summary

1. **Adapters** are modules that let Calcite speak with various data sources (CSV, Cassandra, MongoDB, etc.) as if they were standard SQL tables.
2. They typically expose a **schema** and **tables** (like a DB schema).
3. They can implement **optimizations** to “push down” certain parts of the query (filters, aggregations) to the data source for better performance.
4. You configure them usually via a **JSON model** or Java code.
5. **Writing your own adapter** involves implementing certain Calcite interfaces (schema, table, possibly planner rules).

That’s how Calcite adapters work. They are the **glue** between Calcite’s SQL capabilities and your actual data sources, giving Calcite the power to query almost anything with SQL!