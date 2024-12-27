# Source

https://calcite.apache.org/docs/algebra.html  

Below is a **plain-language** explanation of **relational algebra** in Calcite, why it’s considered “algebra,” and how it connects to mathematics. Afterwards, we’ll briefly explain how Calcite’s “Algebra Builder” works (i.e., the `RelBuilder`), tying it back to these ideas.

------

## 1. What is “Algebra” in Calcite?

### 1.1 High-level Idea

- When we say “Algebra” in **relational algebra**, we’re not talking about high-school algebra with x’s and y’s.
- **Relational algebra** is a formal way of describing how to manipulate data organized in **tables**.
- You can think of each **table** as a “relation,” and the operations (like **SELECT**, **JOIN**, **GROUP BY**) as “algebraic transformations” that take one or more tables as input and produce a new table.

### 1.2 Link to Mathematics

- **Relational algebra** was founded on **set theory** (in mathematics) and **logic**.
- A table (or *relation*) is seen as a set of rows (or *tuples*).
- Operations such as **union**, **intersection**, **projection** (like SELECT certain columns), **selection** (filter rows), and **join** are mathematically defined so they can be **combined** or **rearranged** without changing the final results.
- This means you can safely **rewrite** a query or “push” filters closer to where the data lives, just like you can rearrange algebraic expressions in math (e.g., a+b=b+aa + b = b + a).

### 1.3 Why Calcite Uses Relational Algebra

- A **query optimizer** (like Calcite) will try different equivalent ways (plans) to execute your query.

- It relies on 

  mathematical identities

   from relational algebra. For instance:

  - **Filter** can often move “under” a **join** if it doesn’t affect the other table, which speeds up the query.
  - **Projection** (choosing certain columns) can be pushed “down” to read fewer columns from disk.

- Because these rules come from a well-defined mathematical framework, Calcite can confidently rearrange queries without changing the meaning (semantics).

------

## 2. Calcite’s Planner and Algebra

### 2.1 Expression Trees

- Internally, Calcite converts every SQL query into a 

  tree

   of relational operations.

  - Example: a `SELECT` might become a **Project** node in the tree; a `WHERE` clause might become a **Filter** node.

- These trees are manipulated by **planner rules**—small bits of logic that say, “If you see a Filter on top of a Join, try pushing that Filter inside the Join inputs.”

### 2.2 Cost-based Optimization

- Each alternative “plan” has a “cost” (e.g., how many rows it must process, how expensive the operations are).
- Calcite picks the plan with the lowest cost while preserving correctness—**algebraic laws** guarantee correctness.

------

## 3. Algebra Builder (`RelBuilder`)

### 3.1 Purpose

- The `RelBuilder` is a **Java utility** in Calcite that lets you build these relational algebra trees directly, without writing SQL.

- Instead of “

  ```
  SELECT … FROM X JOIN Y
  ```

  ”, you can do something like:

  ```java
  builder
    .scan("EMP")                // start with table EMP
    .filter(condition)          // apply a filter
    .project(listOfFields)      // project certain columns
    .build();                   // produce the final RelNode
  ```

### 3.2 Stack-based Construction

- `RelBuilder` uses a **stack** internally. Every time you call something like `.scan("EMP")`, it places a new relational expression on the stack.
- Calls like `.filter(...)` or `.join(...)` pop the top 1 or 2 items, apply the operation, and push a new result.

### 3.3 Algebraic Operators

- Each method (`filter`, `project`, `aggregate`, `join`) corresponds to a **relational algebra** operator.
- By chaining them, you build a tree of operations in memory, which is the same structure Calcite would create after parsing SQL.

### 3.4 Why Bother?

- This is handy if you prefer building queries programmatically or you’re writing custom transformations (like adding your own optimization rules).
- The final “tree” you build with `RelBuilder` can be fed into Calcite’s optimizer, which then uses the same **algebraic laws** to optimize it.

------

## 4. Key Takeaways

1. **Relational Algebra and Math**
   - Relational algebra is grounded in **set theory** and **logic**. It is called “algebra” because it provides a collection of **operators** (like mathematical operators) and **identities** that let you transform expressions.
2. **Core of Query Optimization**
   - Calcite uses relational algebra as the “language” of queries under the hood.
   - Planner rules apply valid transformations (like in algebra, a×(b+c)=ab+aca \times (b + c) = ab + ac, but for tables).
3. **`RelBuilder` for Programmatic Construction**
   - If you don’t want to write SQL, you can use `RelBuilder` to create or modify queries in code.
   - This is still the same relational algebra approach—just assembled in Java rather than typed as SQL.
4. **Extensible**
   - You can add your own **operators** (if you have a custom join type), your own **cost model**, or your own **planner rules** if you have domain-specific needs.

------

### In Short

- **Mathematics** (specifically **set theory** and **logic**) underpins **relational algebra**.
- **Relational algebra** underpins **SQL** and query optimization in Calcite.
- **Calcite** uses these algebraic laws to optimize queries by rewriting them into cheaper but equivalent forms.
- The **Algebra Builder** (`RelBuilder`) is just a programmatic way of building these algebraic expressions (i.e., relational operations) instead of writing SQL strings.

That’s how “Algebra” in Calcite relates to mathematics—and why it’s essential for how Calcite processes and optimizes queries.