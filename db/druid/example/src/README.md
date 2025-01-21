# Commands

Generate new project

```bash
mvn archetype:generate -DgroupId=com.alibaba.druid -DartifactId=example
cd example
```

Build

```bash
mvn clean package
```

Run

```bash
mvn exec:java -Dexec.mainClass="com.alibaba.druid.App"
```
