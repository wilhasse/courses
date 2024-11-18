# Guide:

https://github.com/polardb/polardbx-sql/blob/main/docs/en/quickstart-development.md

# PolarDBX Engine

Debian 12

Packages:  

```bash
apt install make automake cmake git bison libaio-dev libncurses-dev libsasl2-dev libldap2-dev libssl-dev pkg-config ligtool
apt install libsnappy-dev libbz2-dev liblz4-dev
```

Compile:

```bash
cmake .     \
-DFORCE_INSOURCE_BUILD=ON    \
-DCMAKE_BUILD_TYPE="Debug"     \
-DSYSCONFDIR="/u01/mysql"     \
-DCMAKE_INSTALL_PREFIX="/u01/mysql"     \
-DMYSQL_DATADIR="/u01/mysql/data"     -DWITH_BOOST="./extra/boost/boost_1_77_0.tar.gz"     \
-DDOWNLOAD_BOOST=1     \
-DCMAKE_CXX_FLAGS="-Wno-error=unused-value"     \
-DWITH_JEMALLOC=ON
```

Error 1:

```txt
[35%] Building CXX object sql/CMakeFiles/master.dir/rpl_source.cc.o
In file included from /home/cslog/polardbx-engine/sql/rpl_binlog_sender.h:40,
 from /home/cslog/polardbx-engine/sql/rpl_source.cc:66:
/home/cslog/polardbx-engine/sql/lizard_rpl_binlog_sender.h:110:14: error: ‘Event’ was not declared in this scope 110 | std::array<Event, 2> m_events;
```

Edit source sql/lizard_rpl_binlog_sender.h

```c
#include <array>
#include <string>
```

Error 2:

```txt
make[2]:  [mysys/CMakeFiles/build_id_test.dir/build.make:116: runtime_output_directory/build_id_test] Erro 1
make[2]:  Apagando arquivo 'runtime_output_directory/build_id_test'
make[1]: ** [CMakeFiles/Makefile2:7431: mysys/CMakeFiles/build_id_test.dir/all] Erro 2
make: * [Makefile:166: all] Erro 2
```

Run make with locale in English

```bash
LC_ALL=C make
```

# PolarDBX SQL

Ubuntu 24-0

Packages:

```bash
sudo apt-get install openjdk-11-jdk openjdk-11-jdk-headless openjdk-11-jre-zero
java -version

sudo apt install maven
mvn -version
```

Repositories:

```bash
git clone https://github.com/polardb/polardbx-sql
git clone https://github.com/polardb/polardbx-glue
git clone https://github.com/polardb/polardbx-cdc
git clone https://github.com/alibaba/canal.git
```

Build:


1) cd polardbx-sql

```bash
git submodule update --init
mvn install -D maven.test.skip=true -D env=release -e
```

Error:  
Could not find artifact com.alibaba.polardbx:polardbx-optimizer:jar:tests:5.4.19-SNAPSHOT

Pom.xml

Add in plugins

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-jar-plugin</artifactId>
    <version>3.1.0</version>
    <executions>
        <execution>
            <goals>
                <goal>test-jar</goal>
            </goals>
        </execution>
    </executions>
</plugin>
```

```bash
cd polardbx-optimizer
mvn install -DskipTests -pl :polardbx-optimizer
cd ..
mvn install -D maven.test.skip=true -D env=release -e
```


2) canal

```bash
cd ~/canal
mvn clean install -DskipTests
```


3) polardbx-cdc

```bash
cd polardbx-cdc/
git submodule update --init
```

Error: change polardbx-parser to version already compiled

```xml
-       <polardbx-parser.version>6.21</polardbx-parser.version>
+       <polardbx-parser.version>5.4.19-SNAPSHOT</polardbx-parser.version>
```


4) polaerdbx-glue

Edit pom.xml to reference dir polardbx-sql

```xml
-        <relativePath>../pom.xml</relativePath>
+        <relativePath>../polardbx-sql/pom.xml</relativePath>
```

mvn install -D maven.test.skip=true -D env=release -e
