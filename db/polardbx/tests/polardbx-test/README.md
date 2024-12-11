# Introduction

PolarDB-X Plugin Java Test

# Run

Build

```bash
mvn clean package
```

Dependency Tree

```bash
mvn dependency:tree
```

Run

Choosing the class

```bash
D:\courses\db\polardbx\tests\polardbx-test>java -jar target/polardbx-test-1.0-SNAPSHOT.jar
Please provide a command number or name:
  1) parsesql     - Run SQL parsing tests
  2) simplequery  - Run simple database query tests
  3) server       - Run simple server
```

Run the class directly using maven

```bash
# compile and run
mvn clean compile exec:java -Dexec.mainClass="SimpleServer" -X

# only run -X full stack trace
mvn exec:java -Dexec.mainClass="SimpleServer" -X
```

# Parse SQL

- SQL Parsing - FastsqlParser to convert SQL Text into a SqlNode (Abstract Syntax Tree)
- Logical Plan Generation: Converts the SqlNode to a RelNode (logical plan)
- Physical Plan Generation: Optimizes the logical plan into a physical execution plan

To integrate with simplequery (PolarDBX RPC):

- Generate the physical plan using this parser
- Convert the physical plan to an ExecPlan protobuf message
- Send it through your existing RPC channel

