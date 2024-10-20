# Install

Star Schema Benchmark (SSB)

Link: https://github.com/pingcap/tidb-bench

```bash
sudo apt-get install build-essential
sudo git clone https://github.com/pingcap/tidb-bench
cd /tidb-bench/ssb/dbgen
make -j8
```

# Generate Data

genererate tlb files

```bash
# "-s" is used to specify the volume of data to generate in GB.
# "-T a" indicates dbgen to generate data for all tables.
./dbgen -s 10 -T a
```

# MySQL

Enable load in file
add to /etc/mysql/conf.d/mysqld.cnf

```ini
local_infile=1
```

# Import Data

```bash
mysql ssb -u root -pXXX --local-infile=1 -e "load data local infile 'supplier.tbl'  into table supplier  fields terminated by '|' lines terminated by '\n';"
mysql ssb -u root -pXXX --local-infile=1 -e "load data local infile 'customer.tbl'  into table customer  fields terminated by '|' lines terminated by '\n';"
mysql ssb -u root -pXXX --local-infile=1 -e "load data local infile 'date.tbl'      into table date      fields terminated by '|' lines terminated by '\n';"
mysql ssb -u root -pXXX --local-infile=1 -e "load data local infile 'lineorder.tbl' into table lineorder fields terminated by '|' lines terminated by '\n';"
```

# Run benchmark

Manual

```bash
 (date;mysql ssb -u root -pXXX < 1.sql > 1_res.txt; date)
 (date;mysql ssb -u root -pXXX < 2.sql > 2_res.txt; date)
 (date;mysql ssb -u root -pXXX < 3.sql > 3_res.txt; date)
```
