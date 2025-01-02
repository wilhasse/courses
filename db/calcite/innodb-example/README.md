# Introduction

Calcite using Innodb Adapter to query SSB Benchmark ibd files

https://calcite.apache.org/docs/innodb_adapter.html  


# Tools

Innodb Java Reader  
https://github.com/alibaba/innodb-java-reader

```bash
git clone https://github.com/alibaba/innodb-java-reader
cd innodb-java-reader
mvn clean install -DskipTests -Dmaven.test.skip=true -Dpmd.skip=true
mkdir courses\db\calcite\innodb-example\lib
copy target\innodb-java-reader-1.0.10.jar courses\db\calcite\innodb-example\lib
```

Note: Only documenting it is not necessary because I added jar file on this project

Sqlline  
https://github.com/julianhyde/sqlline

```bash
git clone https://github.com/julianhyde/sqlline
cd sqlline
mvn clean install
# Windows: add environment path
# D:\sqlline-sqlline-1.12.0\bin
```

# Example

Edit src/main/resources/ssb-model.json, adjust create table sql file and data:

```xml
        "sqlFilePath": ["/home/cslog/courses/db/calcite/innodb-example/src/main/resources/ssb-tables.sql"],
        "ibdDataFileBasePath": "/data/polardb/ssb"
```

```bash
# compile
 cd courses/db/calcite/innodb-example/
mvn clean install

# run code
java -cp target/calcite-innodb-example-1.0-SNAPSHOT-jar-with-dependencies.jar SSBQueryExample
Executing query:
SELECT sum("lineorder"."lo_extendedprice" * "lineorder"."lo_discount") AS "revenue"
FROM "lineorder", "date"
WHERE "lineorder"."lo_orderdate" = "date"."d_datekey"
AND "date"."d_year" = 1993
AND "lineorder"."lo_discount" between 1 and 3
AND "lineorder"."lo_quantity" < 25
revenue: 446268068091

Executing query:
SELECT COUNT(*) AS "count" FROM "lineorder"
count: 6001171

# run queries in sqlline
# It still needs jar dependencies in innodb-example jar file
# like calcite jdbc driver and innodb-java-reader 
D:\courses\db\calcite\innodb-example>java -cp target/calcite-innodb-example-1.0-SNAPSHOT-jar-with-dependencies.jar sqlline.SqlLine
sqlline version 1.12.0
sqlline> !connect jdbc:calcite:model=src\main\resources\ssb-model.json admin admin
Transaction isolation level TRANSACTION_REPEATABLE_READ is not supported. Default (TRANSACTION_NONE) will be used instead.
0: jdbc:calcite:model=src\main\resources\ssb->
0: jdbc:calcite:model=src\main\resources\ssb-> SELECT * FROM "customer" LIMIT 5;
```

# Jdk

I got an error complaining about javadoc

```bash
sudo apt-get install default-jdk
update-alternatives --config java
0            /usr/lib/jvm/java-17-openjdk-amd64/bin/java   1711      modo automático
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

I also got an error about:

```bash
[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin:3.8.1:compile (default-compile) on project calcite-innodb-example: Fatal error compiling: java.lang.IllegalAccessError: class lombok.javac.apt.LombokProcessor (in unnamed module @0x517a2b0) cannot access class 
```

I added pom.xml

```xml
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
    <version>1.18.30</version>
</dependency>
```