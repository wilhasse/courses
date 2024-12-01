# MYSQL41

Old native mysql authentication

```bash
-- Drop the user to start fresh
DROP USER IF EXISTS 'teste'@'%';

-- Create the user with explicit authentication method
CREATE USER 'teste'@'%' IDENTIFIED WITH mysql_native_password BY 'teste';

-- Grant privileges
GRANT ALL PRIVILEGES ON *.* TO 'teste'@'%' WITH GRANT OPTION;
```

Checking the type

```bash
mysql> SELECT user, host, plugin FROM mysql.user WHERE user = 'teste';
+-------+------+-----------------------+
| user  | host | plugin                |
+-------+------+-----------------------+
| teste | %    | mysql_native_password |
+-------+------+-----------------------+
1 row in set (0,01 sec)
```

# sha256_password

Deprecated in MySQL 8 will be removed in future release

```bash
-- Drop the user to start fresh
DROP USER IF EXISTS 'teste'@'%';

-- Create the user with explicit authentication method
CREATE USER 'teste'@'%' IDENTIFIED WITH sha256_password BY 'teste';

-- Grant privileges
GRANT ALL PRIVILEGES ON *.* TO 'teste'@'%' WITH GRANT OPTION;
```

Checking the type

```bash
mysql> SELECT user, host, plugin FROM mysql.user WHERE user = 'teste';
+-------+------+-----------------+
| user  | host | plugin          |
+-------+------+-----------------+
| teste | %    | sha256_password |
+-------+------+-----------------+
1 row in set (0,00 sec)
```

# caching_sha2_password

Default authentication plugin in MySQL 8.0.4 and above  

Doc: https://dev.mysql.com/doc/mysql-security-excerpt/8.0/en/caching-sha2-pluggable-authentication.html  

```bash
-- Drop the user to start fresh
DROP USER IF EXISTS 'teste'@'%';

-- Create the user with explicit authentication method
CREATE USER 'teste'@'%';

-- Grant privileges
GRANT ALL PRIVILEGES ON *.* TO 'teste'@'%' WITH GRANT OPTION;

-- Change Password
SET password for teste@'%'='teste';
```

Checking the type

```bash
mysql> SELECT user, host, plugin FROM mysql.user WHERE user = 'teste';
+-------+------+-----------------------+
| user  | host | plugin                |
+-------+------+-----------------------+
| teste | %    | caching_sha2_password |
+-------+------+-----------------------+
1 row in set (0,01 sec)
```

You can use SET password with other plugin but have to change de default in my.ini example:

```ini
[mysqld]
default_authentication_plugin=sha256_password
```