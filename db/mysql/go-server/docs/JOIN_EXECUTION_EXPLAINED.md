# How JOIN Execution Works in go-mysql-server

## Overview

When you execute a join query like:
```sql
SELECT product_id 
FROM users u, orders o 
WHERE u.id = o.user_id AND u.id = 1;
```

go-mysql-server performs several steps to execute this query efficiently.

## Step-by-Step Execution

### 1. Query Parsing
The SQL string is parsed into an Abstract Syntax Tree (AST) using the Vitess parser.

### 2. Query Analysis and Optimization

The analyzer applies multiple optimization rules:

#### a) Cross Join to Inner Join
```
FROM users u, orders o WHERE u.id = o.user_id
```
Is converted to:
```
FROM users u INNER JOIN orders o ON u.id = o.user_id
```

#### b) Filter Pushdown
The filter `u.id = 1` is pushed down to the users table scan, so only user with id=1 is read.

#### c) Join Reordering
The optimizer may reorder tables based on estimated row counts.

### 3. Execution Plan

The final execution plan looks like:
```
Project(product_id)
 └─ Filter(u.id = o.user_id AND u.id = 1)
     └─ CrossProduct
         ├─ Table(users)
         └─ Table(orders)
```

After optimization:
```
Project(product_id)
 └─ InnerJoin(u.id = o.user_id)
     ├─ Filter(id = 1)
     │   └─ Table(users)
     └─ Table(orders)
```

### 4. Join Algorithm

go-mysql-server uses different join algorithms:

#### Nested Loop Join (Default)
```
for each row in left_table:
    for each row in right_table:
        if join_condition matches:
            output combined row
```

#### Hash Join (For Equi-joins)
```
1. Build phase: Create hash table from smaller table
2. Probe phase: For each row in larger table, lookup in hash table
```

### 5. Row-by-Row Execution

For your query with the data:
- Users: {id: 1, name: "Alice"}, {id: 2, name: "Bob"}
- Orders: {id: 1, user_id: 1, product_id: 1}, {id: 2, user_id: 1, product_id: 4}

Execution flow:
1. Scan users table with filter id=1 → Returns Alice
2. For Alice (id=1), scan orders table
3. Find orders where user_id=1 → Returns 2 orders
4. Project product_id from each matched row → Returns [1, 4]

But you got 4 rows because you have duplicate data in your users table!

## Understanding the Debug Output

When you run the debug server with your query, you'll see:

```
INFO: 🔍 Looking up table                    table=users
INFO: 🔍 Looking up table                    table=orders
INFO: Optimizer rule: replaceCrossJoins converting comma join to INNER JOIN
INFO: Optimizer rule: pushFilters pushing u.id = 1 to users table
INFO: 📊 Starting table scan                 table=users
INFO: 📄 Reading row                         data="[1 Alice ...]"
INFO: 📊 Starting table scan                 table=orders
INFO: 📄 Reading row                         data="[1 1 1 ...]"  // user_id=1, product_id=1
INFO: 📄 Reading row                         data="[2 1 4 ...]"  // user_id=1, product_id=4
```

## Join Optimization Rules

1. **replaceCrossJoins**: Converts comma joins to proper JOINs
2. **pushFilters**: Moves WHERE conditions to table scans
3. **optimizeJoins**: Reorders joins based on statistics
4. **applyHashIn**: Converts IN clauses to hash lookups
5. **pruneColumns**: Removes unnecessary columns early

## Performance Considerations

1. **Filter Early**: Push filters to reduce rows scanned
2. **Small Table First**: Join smaller result sets first
3. **Use Indexes**: Indexed columns speed up joins
4. **Hash Joins**: Better for large equi-joins
5. **Avoid Cartesian Products**: Always have join conditions

## Debugging Join Performance

Enable analyzer debug mode to see:
- Which optimization rules fire
- How the query plan changes
- Estimated vs actual row counts
- Which join algorithm is chosen

The debug server shows you exactly how many rows are read from each table and in what order, making it easy to understand join performance.