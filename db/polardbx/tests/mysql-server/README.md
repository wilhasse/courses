# Introduction

MySQL Server test

# Run

Build

```bash
mvn clean package
```

Run

```bash
java -jar target/simple-mysql-server-1.0-SNAPSHOT-jar-with-dependencies.jar
```

Client

```bash
mysql -h127.0.0.1 -uroot -p --ssl-mode=DISABLED --enable-cleartext-plugin
```
