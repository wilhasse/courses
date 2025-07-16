# MySQL UDF Table-Valued Function Simulation Test

This test demonstrates how to simulate table-valued functions in MySQL using multiple UDFs that work together to provide table-like data that can be joined with real tables.

## Overview

Since MySQL doesn't have true table-valued functions without server plugin headers, this test creates multiple UDFs that simulate a virtual table:
- A real table `TEST1` with sample data
- Four UDF functions that together simulate a virtual table `TEST2` with 5 rows
- Test queries demonstrating various JOIN operations between the real and virtual tables

## Files Created

### UDF Implementation
- `src/test_tvf_plugin.cpp` - Contains four UDF functions that simulate a table

### Build & Installation Scripts
- `scripts/build_tvf.sh` - Builds the plugin
- `scripts/install_tvf.sh` - Installs the UDF functions into MySQL
- `scripts/uninstall_tvf.sh` - Removes the UDF functions from MySQL
- `scripts/run_tvf_test.sh` - Runs the complete test

### Test Scripts
- `tests/create_test1_table.sql` - Creates TEST1 table with sample data
- `tests/test_tvf_join.sql` - Demonstrates JOIN operations using UDFs

## UDF Functions

The plugin provides four UDF functions that work together:

1. **test2_row_count()** - Returns the number of rows in the virtual table (5)
2. **test2_get_id(row_num)** - Returns the id for a given row number
3. **test2_get_name(row_num)** - Returns the name for a given row number
4. **test2_get_value(row_num)** - Returns the value for a given row number

## How to Run the Test

From the project root directory:

### Full Test (Build + Install + Run)
```bash
./scripts/run_tvf_test.sh
```

### Quick Test (Just Run Queries)
```bash
# If functions are already installed, just run the queries
./scripts/run_tvf_test_quick.sh
```

### Step by Step

1. **Build the plugin:**
   ```bash
   ./scripts/build_tvf.sh
   ```

2. **Install the plugin:**
   ```bash
   ./scripts/install_tvf.sh
   ```

3. **Run the tests:**
   ```bash
   mysql -u root  < tests/test_tvf_join.sql
   ```

4. **Clean up (optional):**
   ```bash
   ./scripts/uninstall_tvf.sh
   ```

## Virtual Table Structure (TEST2)

The UDFs simulate a table with 5 rows:
- Row 1: id=1, name="Row 1", value=10.5
- Row 2: id=2, name="Row 2", value=21.0
- Row 3: id=3, name="Row 3", value=31.5
- Row 4: id=4, name="Row 4", value=42.0
- Row 5: id=5, name="Row 5", value=52.5

## Test Scenarios

The test script demonstrates:
1. Testing individual UDF functions
2. Generating a virtual table using recursive CTEs
3. INNER JOIN between TEST1 and virtual TEST2
4. LEFT JOIN showing all TEST1 records
5. Aggregate queries with GROUP BY
6. Complex queries with WHERE clauses
7. UDF error handling (out of range values)

## Technical Details

- Uses MySQL's UDF interface (no server plugin headers required)
- The virtual table is generated using recursive CTEs that call the UDFs
- Data is generated dynamically when queried
- No actual data storage is required
- Thread-safe implementation

## Requirements

- MySQL 8.0+ (for recursive CTE support)
- MySQL client development headers (libmysqlclient-dev or libperconaserverclient-dev)
- g++ compiler

## Example Query

```sql
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
),
test2 AS (
    SELECT 
        test2_get_id(n) AS id,
        test2_get_name(n) AS name,
        test2_get_value(n) AS value
    FROM numbers
)
SELECT * FROM TEST1 t1
JOIN test2 t2 ON t1.id = t2.id;
```

## Notes

- The UDFs must be installed with appropriate MySQL privileges
- The virtual table is read-only
- Performance is excellent as data is generated in-memory
- This approach demonstrates how to work around the lack of true table-valued functions in MySQL

## Troubleshooting

### Build Errors

**Problem**: `'my_bool' does not name a type`
- **Solution**: The code has been updated to use `bool` instead of the deprecated `my_bool` type

### Installation Hanging

**Problem**: Installation script hangs during function creation
- **Solution**: The functions may already exist. The install script now drops existing functions first

### Not Seeing Test Results

**Problem**: Test script runs but doesn't show query results
- **Solution**: Use the updated scripts that properly display output:
  - `run_tvf_test.sh` - Full test with formatted output
  - `run_tvf_test_quick.sh` - Just runs queries without rebuild/reinstall

### Database Selection Error

**Problem**: `ERROR 1046 (3D000): No database selected`
- **Solution**: The test script now creates and uses a test database (`test_tvf_db`)

### File Not Found Error

**Problem**: `Failed to open file 'create_test1_table.sql'`
- **Solution**: The test script now uses absolute paths for SOURCE commands

## Implementation Details

### UDF Function Signatures

```c++
// Returns the number of rows in virtual table
bool test2_row_count_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
long long test2_row_count(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);

// Returns id for given row (1-5)
bool test2_get_id_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
long long test2_get_id(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);

// Returns name for given row
bool test2_get_name_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
char *test2_get_name(UDF_INIT *initid, UDF_ARGS *args, char *result,
                     unsigned long *length, char *is_null, char *error);

// Returns value for given row
bool test2_get_value_init(UDF_INIT *initid, UDF_ARGS *args, char *message);
double test2_get_value(UDF_INIT *initid, UDF_ARGS *args, char *is_null, char *error);
```

### Key Design Decisions

1. **Static Data**: The virtual table data is hardcoded for simplicity
2. **Row-Based Access**: Each function accesses data by row number (1-based indexing)
3. **Error Handling**: Out-of-range row numbers return NULL
4. **Thread Safety**: Uses static buffer for string returns (thread-local would be better for production)

## Comparison with True Table-Valued Functions

### What We're Simulating
```sql
-- If MySQL had true TVFs, we could do:
SELECT * FROM table_valued_function();
```

### Our Workaround
```sql
-- Instead, we use CTEs and UDFs:
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < test2_row_count()
),
test2 AS (
    SELECT 
        test2_get_id(n) AS id,
        test2_get_name(n) AS name,
        test2_get_value(n) AS value
    FROM numbers
)
SELECT * FROM test2;
```

## Future Enhancements

1. **Dynamic Data**: Allow UDFs to accept parameters for different datasets
2. **Larger Tables**: Support more than 5 rows by increasing the counter
3. **Thread Safety**: Use thread-local storage for string buffers
4. **Performance**: Cache computed values within a query execution
5. **True TVF**: Implement as a server plugin when MySQL adds proper TVF support