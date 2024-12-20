# Config

Save to a file: qatest.properties
Place it to polardbx-test\src\test\resources

```bash
# PolarDB-X Connection Settings
polardbxUserName=polardbx_root
polardbxPassword=MqAtigzR
polardbxPort=62671
polardbxAddr=10.1.1.121

# Meta DB Connection Settings
metaDbName=polardbx_meta_db
metaDbUser=admin
metaDbPasswd=rimDcghy
metaPort=14407
metaDbAddr=10.1.1.132

# MySQL Settings (same as PolarDB-X for testing)
mysqlUserName=polardbx_root
mysqlPassword=MqAtigzR
mysqlPort=62671
mysqlAddr=10.1.1.121

# Database Names
polardbxDb=drds_polarx1_qatest_app
polardbxDb2=drds_polarx2_qatest_app
polardbxNewDb=drds_polarx1_part_qatest_app
polardbxNewDb2=drds_polarx2_part_qatest_app
archiveDb=archived1
archiveDb2=archived2
mysqlDb=andor_qatest_polarx1
mysqlDb2=andor_qatest_polarx2

# Connection Properties
connProperties=allowMultiQueries=true&rewriteBatchedStatements=true&characterEncoding=utf-8

# Test Configurations
strictTypeTest=false
use_file_storage=faalse
columnar_mode=false
skip_create_columnar_index=false
enableAsyncDDL=true
useDruid=false
engine=oss

# Cluster Configuration
dnCount=1
shardDbCountEachDn=4

# Test Parameters
transferTestTime=1
allTypesTestPrepareThreads=16
allTypesTestBigColumn=false
transferRowCount=100
```

# Tables

```sql
-- Database
CREATE DATABASE IF NOT EXISTS drds_polarx1_qatest_app;
USE drds_polarx1_qatest_app;

-- Base table with single index
CREATE TABLE gsi_dml_test_unique_one_index_base (
  pk INT NOT NULL,
  integer_test INT,
  bigint_test BIGINT,
  varchar_test VARCHAR(255),
  datetime_test DATETIME,
  year_test YEAR,
  char_test CHAR(255),
  PRIMARY KEY (pk),
  UNIQUE KEY idx1 (integer_test)
) dbpartition by hash(pk) tbpartition by hash(pk) tbpartitions 4;

-- Create GSI for one_index base
CREATE GLOBAL INDEX gsi_dml_test_unique_one_index_index1 ON gsi_dml_test_unique_one_index_base (integer_test) 
COVERING (bigint_test, varchar_test) dbpartition by hash(integer_test) tbpartition by hash(integer_test) tbpartitions 4;

-- Now the multi-index table
CREATE TABLE gsi_dml_test_unique_multi_index_base (
  pk INT NOT NULL,
  integer_test INT,
  bigint_test BIGINT,
  varchar_test VARCHAR(255),
  datetime_test DATETIME,
  year_test YEAR,
  char_test CHAR(255),
  PRIMARY KEY (pk),
  UNIQUE KEY idx1 (integer_test),
  UNIQUE KEY idx2 (bigint_test)
) dbpartition by hash(pk) tbpartition by hash(pk) tbpartitions 4;

-- Create GSIs for multi-index base
CREATE GLOBAL INDEX gsi_dml_test_unique_multi_index_index1 ON gsi_dml_test_unique_multi_index_base (integer_test)
COVERING (bigint_test, varchar_test) dbpartition by hash(integer_test) tbpartition by hash(integer_test) tbpartitions 4;

CREATE GLOBAL INDEX gsi_dml_test_unique_multi_index_index2 ON gsi_dml_test_unique_multi_index_base (bigint_test)
COVERING (integer_test, varchar_test) dbpartition by hash(bigint_test) tbpartition by hash(bigint_test) tbpartitions 4;
```

# Run

```bash
mvn test -Dtest=com.alibaba.polardbx.qatest.dml.sharding.gsi.DeleteGsiErrorTest -DfailIfNoTests=false

```
