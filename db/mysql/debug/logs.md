# Logs

Macros to log in mysql_error.log

```c
#define sql_print_information(...) \
  log_errlog_formatted(INFORMATION_LEVEL, ##__VA_ARGS__)

#define sql_print_warning(...) \
  log_errlog_formatted(WARNING_LEVEL, ##__VA_ARGS__)

#define sql_print_error(...) log_errlog_formatted(ERROR_LEVEL, ##__VA_ARGS__)
```

Where to find the logs:

```bash
mysql> SHOW VARIABLES LIKE 'log_error';
+---------------+------------------------------+
| Variable_name | Value                        |
+---------------+------------------------------+
| log_error     | /data/my3306/mysql_error.log |
+---------------+------------------------------+
1 row in set (0,00 sec)
```

Verbosity, what to log?

```bash
mysql> SHOW VARIABLES LIKE 'log_error_verbosity';
+---------------------+-------+
| Variable_name       | Value |
+---------------------+-------+
| log_error_verbosity | 3     |
+---------------------+-------+
1 row in set (0,00 sec)
```

Values

- Level 1 (ERROR):

SET GLOBAL log_error_verbosity = 1;

Only logs errors
Most critical issues that need immediate attention  
Example: Connection failures, syntax errors, crashed tables  


- Level 2 (ERROR + WARNING):

SET GLOBAL log_error_verbosity = 2;

Logs both errors and warnings  
Warnings are potential issues that didn't prevent operation but might indicate problems  
Example: Using deprecated features, implicit type conversions  


- Level 3 (ERROR + WARNING + INFORMATION):

SET GLOBAL log_error_verbosity = 3;

Logs errors, warnings, and informational messages  
Includes notes about query execution, system status changes  
Your sql_print_information messages will only appear at this level 