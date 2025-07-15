# JSON_TABLE Tips and Troubleshooting

## Common Issues and Solutions

### 1. "Invalid JSON text" Error

**Problem**: 
```sql
/* Erro SQL (3141): Invalid JSON text in argument 1 to function json_table: "Invalid value." at position 0. */
```

**Causes**:
- Response too large (over 1MB limit)
- Empty result set
- Query error

**Solution**:
Always add LIMIT to your queries and check the raw JSON first:

```sql
-- Check raw JSON output
SELECT CAST(chdb_api_query_json('YOUR QUERY HERE LIMIT 10') AS CHAR)\G

-- If it shows "ERROR: Response too large", reduce the LIMIT
-- If it shows an error message, fix the query
```

### 2. Single Column Queries

Working example:
```sql
SELECT jt.* 
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('SELECT DISTINCT id_contr FROM mysql_import.historico WHERE codigo=22 LIMIT 100') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr'
    )
) AS jt;
```

### 3. Multiple Columns

```sql
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('SELECT id_contr, seq, codigo FROM mysql_import.historico LIMIT 100') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;
```

### 4. Handling Large Result Sets

For queries that might return large results:

```sql
-- Option 1: Use aggregation in ClickHouse
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT codigo, COUNT(*) as cnt 
        FROM mysql_import.historico 
        GROUP BY codigo 
        ORDER BY cnt DESC 
        LIMIT 50
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        codigo INT PATH '$.codigo',
        cnt INT PATH '$.cnt'
    )
) AS jt;

-- Option 2: Use pagination
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT * FROM mysql_import.historico 
        WHERE id_contr > 1000000 
        ORDER BY id_contr 
        LIMIT 100 OFFSET 0
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr',
        seq INT PATH '$.seq',
        codigo INT PATH '$.codigo'
    )
) AS jt;
```

### 5. Debugging Tips

1. **Always test the raw JSON first**:
   ```sql
   SELECT CAST(chdb_api_query_json('YOUR QUERY') AS CHAR)\G
   ```

2. **Check data size**:
   ```sql
   -- Count records before fetching
   SELECT CAST(chdb_api_query('SELECT COUNT(*) FROM mysql_import.historico WHERE codigo=22') AS CHAR);
   ```

3. **Use LIMIT liberally**:
   - Start with `LIMIT 10` for testing
   - Increase gradually based on data size
   - Remember the 1MB response limit

### 6. Column Name Case Sensitivity

ClickHouse column names are typically lowercase:
```sql
-- Correct
SELECT id_contr FROM ...

-- Incorrect (might fail)
SELECT ID_CONTR FROM ...
```

### 7. Complex Data Types

For complex ClickHouse types:
```sql
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT 
            toDate(data) as date,
            toString(status) as status,
            toFloat64(valor) as value
        FROM mysql_import.historico 
        LIMIT 10
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        date DATE PATH '$.date',
        status VARCHAR(50) PATH '$.status',
        value DECIMAL(10,2) PATH '$.value'
    )
) AS jt;
```

## Best Practices

1. **Always use LIMIT** in your ClickHouse queries
2. **Test with small datasets first** (LIMIT 10)
3. **Check raw JSON output** when debugging
4. **Use aggregation** to reduce data size
5. **Consider creating VIEWs** for frequently used queries
6. **Monitor response size** to stay under 1MB limit

## Example: Creating a VIEW

```sql
CREATE OR REPLACE VIEW v_active_contracts AS
SELECT jt.*
FROM JSON_TABLE(
    CONVERT(chdb_api_query_json('
        SELECT DISTINCT id_contr 
        FROM mysql_import.historico 
        WHERE codigo IN (22, 767, 13) 
        ORDER BY id_contr 
        LIMIT 1000
    ') USING utf8mb4),
    '$.data[*]' COLUMNS (
        id_contr INT PATH '$.id_contr'
    )
) AS jt;

-- Use the view
SELECT * FROM v_active_contracts WHERE id_contr > 1000000;
```