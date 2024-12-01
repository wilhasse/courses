# Introduction

Select XProtocol Test
Code from this Alibaba's article:  
https://www.alibabacloud.com/blog/an-interpretation-of-polardb-x-source-codes-7-life-of-private-protocol-connection-cn_599458

You can follow along to understand better the X Protocol

# Run

Build

```bash
mvn clean package
```

Run

```bash
java -cp "target/select-test-1.0-SNAPSHOT.jar;D:/polardbx/polardbx-sql/polardbx-rpc/target/polardbx-rpc-5.4.19-SNAPSHOT.jar;D:/polardbx/polardbx-sql/polardbx-common/target/polardbx-common-5.4.19-SNAPSHOT.jar" GalaxyTest
```
