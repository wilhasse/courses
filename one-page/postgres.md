# Link

Download  
https://www.postgresql.org/download/

Duckdb Extension  
https://github.com/duckdb/pg_duckdb

## Install

Postgres 16 in Debian

```baah
# Import the repository signing key:
sudo apt install curl ca-certificates
sudo install -d /usr/share/postgresql-common/pgdg
sudo curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc

# Create the repository configuration file:
sudo sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

# Update the package lists:
sudo apt update

# Install the latest version of PostgreSQL:
# If you want a specific version, use 'postgresql-16' or similar instead of 'postgresql'
sudo apt -y install postgresql
```

## Install Duckdb

```bash
git clone https://github.com/duckdb/pg_duckdb
apt-get install postgresql-server-dev-16
apt-get install lz4 liblz4-dev
make install
```

## Install Pgloader

```bash
git clone https://github.com/dimitri/pgloader
cd pgloader
sudo make install
```

## Install Columnar Hyda

```bash
sudo apt-get install libzstd-dev
git clone https://github.com/hydradatabase/hydra.git
cd hydra/columnar
./configure
make install
```

Verify

```bash
postgres=# SELECT * FROM pg_available_extensions WHERE name = 'columnar';
   name   | default_version | installed_version |         comment
----------+-----------------+-------------------+--------------------------
 columnar | 11.1-12         |                   | Hydra Columnar extension
(1 linha)

postgres=# CREATE EXTENSION columnar;
CREATE EXTENSION

postgres=# SELECT * FROM pg_extension WHERE extname = 'columnar';
  oid  | extname  | extowner | extnamespace | extrelocatable | extversion | extconfig | extcondition
-------+----------+----------+--------------+----------------+------------+-----------+--------------
 17006 | columnar |       10 |         2200 | f              | 11.1-12    |           |
(1 linha)
```

# Basic commands

Login as root

```bash
sudo -u postgres psql
```

New User and Database

```bash
CREATE ROLE cslog LOGIN PASSWORD 'your_password';
CREATE DATABASE newdb OWNER cslog;
```

```bash
psql -U cslog -d newdb

# already user
psql -d siscom
```

Create user and permission for pgloader

```sql
CREATE USER pgloader WITH PASSWORD 'pgloader1';
ALTER USER pgloader WITH CREATEDB;
GRANT CONNECT ON DATABASE siscom TO pgloader;
\c siscom

GRANT USAGE ON SCHEMA public TO pgloader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pgloader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pgloader;
```

Enable external connection

```bash
#change /etc/postgresql/16/main/postgresql.conf
#Find the line starting with listen_addresses and change it to:
listen_addresses = '*'

#add source network to /etc/postgresql/16/main/pg_hba.conf
host    all             all             192.168.10.0/24         md5
```
