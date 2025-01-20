# Concepts

[Main concepts](./concepts.md)

[Protocol](./protocol.md)

# Install

Get last version in Git:  
https://github.com/alibaba/canal/releases

wget https://github.com/alibaba/canal/releases/download/canal-1.1.8/canal.deployer-1.1.8.tar.gz

Doc:  
https://github.com/alibaba/canal/wiki/QuickStart

```bash
mkdir alibaba-canal
cd alibaba-canal
tar xvzf ../canal.deployer-1.1.8.tar.gz
```

# Source

Permission

```bash
CREATE USER canal IDENTIFIED BY 'password';  
GRANT SELECT, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'canal'@'%';
#GRANT ALL PRIVILEGES ON *.* TO 'canal'@'%' ;
FLUSH PRIVILEGES;
```

Config

conf/example/instance.properties

```bash
canal.instance.dbUsername=USER
canal.instance.dbPassword=PASSWORD
canal.instance.connectionCharset = ISO-8859-1
```

# Start

```bash
./bin/startup.sh
```

logs:
```bash
cat logs/example/example.log
2025-01-18 15:47:14.237 [main] INFO  com.alibaba.otter.canal.deployer.CanalLauncher - ## set default uncaught exception handler
2025-01-18 15:47:14.244 [main] INFO  com.alibaba.otter.canal.deployer.CanalLauncher - ## load canal configurations
2025-01-18 15:47:14.251 [main] INFO  com.alibaba.otter.canal.deployer.CanalStarter - ## start the canal server.
2025-01-18 15:47:14.278 [main] INFO  com.alibaba.otter.canal.deployer.CanalController - ## start the canal server[192.168.0.51(192.168.0.51):11111]
2025-01-18 15:47:15.273 [main] INFO  com.alibaba.otter.canal.deployer.CanalStarter - ## the canal server is running now ......
```

# Error

In version 1.1.8 I got an error:

```txt
Value too long for column "CHARACTER VARYING"
```

Download newer version h2 and substitute in canal:

```bash
# 1. Stop canal
./bin/stop.sh

# 2. Go to lib directory
cd lib

# 3. Check current h2 jar version
ls -l h2-*.jar

# 4. Download new H2 jar version 2.2.224
# You can download it from: https://repo1.maven.org/maven2/com/h2database/h2/2.2.224/h2-2.2.224.jar

# 5. Remove old H2 jar
rm h2-2.1.210.jar

# 6. Copy new jar to lib directory
cp h2-2.2.224.jar ./

# 7. Clean old meta files (important!)
cd ../
rm -rf conf/example/meta.dat
rm -rf conf/example/h2.mv.db
rm -rf conf/example/h2.trace.db

# 8. Start canal
./bin/startup.sh
```

# Client

Compile

```bash
mvn clean compile
```
Run example:

```bash
mvn exec:java -Dexec.mainClass="com.alibaba.otter.canal.sample.SimpleCanalClientExample"
```

# Automatic startup

```bash
nano /etc/systemd/system/canal.service

[Unit]
Description=Alibaba Canal Service
After=network.target

[Service]
Type=forking
User=cslog
Group=cslog
Environment="JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64"  # Adjust this path
WorkingDirectory=/home/cslog/alibaba-canal
ExecStart=/home/cslog/alibaba-canal/bin/startup.sh
ExecStop=/home/cslog/alibaba-canal/bin/stop.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable canal
sudo systemctl start canal
sudo systemctl status canal
```