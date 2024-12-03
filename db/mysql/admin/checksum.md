# Introduction

Compare source and replicas


# Percona Toolkit

```bash
apt-get install percona-toolkit
```

# Permission

Add permission to source

```bash
CREATE USER IF NOT EXISTS 'sysadm'@'localhost';
ALTER USER 'sysadm'@'localhost' IDENTIFIED WITH 'mysql_native_password' REQUIRE NONE PASSWORD EXPIRE DEFAULT ACCOUNT UNLOCK;
GRANT ALL PRIVILEGES ON `percona`.* TO 'sysadm'@'localhost';
GRANT DELETE, INSERT, PROCESS, REPLICATION SLAVE, SELECT, SUPER, UPDATE ON *.* TO 'sysadm'@'localhost';
SET PASSWORD FOR sysadm@localhost = '';
```

Add permission to all replicas

SOURCE - Change to IP. Source host where you will run pt-table-checksum

```bash
CREATE USER IF NOT EXISTS 'sysadm'@'$$SOURCE';
GRANT SUPER, PROCESS, REPLICATION SLAVE, SELECT ON *.* TO 'sysadm'@'$$SOURCE';
SET PASSWORD FOR sysadm@'$$SOURCE' = '';
```

# Check

```bash
# create database if is the first time
mysqladmin create percona -u root -p

# compare to find differences
pt-table-checksum -u sysadm --ask-pass --databases=db --no-check-binlog-format --no-check-replication-filters  >& pt_checksum.txt

# alternative: print all the changes in insert format
pt-table-sync --print --replicate percona.checksums localhost -u sysadm >& pt_print.txt
```

# Syncronize

```bash
# correct all the differences
pt-table-sync --execute --replicate percona.checksums localhost -u sysadm
```
