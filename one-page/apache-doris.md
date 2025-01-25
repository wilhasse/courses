# Install

Guide  
https://doris.apache.org/docs/gettingStarted/quick-start

Binary x64  
wget https://apache-doris-releases.oss-accelerate.aliyuncs.com/apache-doris-3.0.3-bin-x64.tar.gz

Java

```bash
apt-get install openjdk-17-jdk openjdk-17-jre
```

## Environment

```bash
pico /etc/security/limits.conf 
* soft nofile 1000000
* hard nofile 1000000

# commit
ulimit -n 655350
```

```bash
cat >> /etc/sysctl.conf << EOF
vm.max_map_count = 2000000
EOF

# take effect immediately
sysctl -p
```

# fe (frontend)

```bash
cd /doris
pico ./fe/conf/fe.conf

JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
priority_networks = 192.168.0.0/16
```

Run

```bash
# bg
./fe/bin/start_fe.sh --daemon

# test
mysql -uroot -P9030 -h127.0.0.1 -e "show frontends;"
```

# be (backend)

```bash
cd /doris
pico ./be/conf/be.conf

JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
priority_networks = 192.168.0.0/16
```

Configure

```bash
# bg
./be/bin/start_be.sh --daemon

# test
mysql -uroot -P9030 -h127.0.0.1 -e "show frontends;"
```

Run

```bash
# I don't want to disable swap so I commented out swap check in start_be.sh

# start
./be/bin/start_be.sh --daemon
```

# Connect

Connect fe to be

```bash
mysql -uroot -P9030 -h127.0.0.1

>ALTER SYSTEM ADD BACKEND "127.0.0.1:9050";
>show backends;

#check Alive: true
```

Change root password

```bash
SET PASSWORD FOR 'root' = PASSWORD('your_new_password');

# or
ALTER USER 'root'@'%' IDENTIFIED BY '';
```

Create new user

```bash
CREATE USER 'appl_canal'@'192.168.10.%' IDENTIFIED BY 'PASSWORD';
GRANT ALL ON *.* TO 'appl_canal'@'192.168.10.%';
```

# Config

Front end memory config is 8G:

./fe/conf/fe.conf

```bash
# For jdk 8
JAVA_OPTS="... -Xmx8192m ..."
# For jdk 17
JAVA_OPTS_FOR_JDK_17="... -Xmx8192m -Xms8192m ..."
```

Depending on the server memory you can increase or decrease


nano /etc/systemd/system/doris-be.service

```bash
[Unit]
Description=Doris Backend Service
After=network.target

[Service]
Type=forking
User=root
Group=root
LimitNOFILE=655350
Environment="JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"
WorkingDirectory=/doris
ExecStart=/doris/be/bin/start_be.sh --daemon
ExecStop=/doris/be/bin/stop_fe.sh
Restart=always

[Install]
WantedBy=multi-user.target
(END)

sudo systemctl daemon-reload
sudo systemctl enable doris-be
sudo systemctl start doris-be
sudo systemctl status doris-be
```

nano /etc/systemd/system/doris-fe.service

```bash
[Unit]
Description=Doris Frontend Service
After=network.target

[Service]
Type=forking
User=root
Group=root
LimitNOFILE=655350
Environment="JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64"
WorkingDirectory=/doris
ExecStart=/doris/fe/bin/start_fe.sh --daemon
ExecStop=/doris/fe/bin/stop_fe.sh
Restart=always

[Install]
WantedBy=multi-user.target

sudo systemctl daemon-reload
sudo systemctl enable doris-fe
sudo systemctl start doris-fe
sudo systemctl status doris-fe
```