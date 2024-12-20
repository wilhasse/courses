# Introduction

How to copy a mysql database on-line and off-line


# Percona Xtrabackup 8

Origin
Note: encrypted based (if not remove --keyring-file)

```bash
# perform copy
# IP_DESTINATION - ip address
# -w 10 - time to wait after that time without receiving data it will close the connection
xtrabackup --backup --stream=xbstream --keyring-file-data=/var/lib/mysql-keyring/keyring-encrypted -u root -p | pigz -c | nc -w 10 $IP_DESTINATION 9999
```

Destination

Note: if encrypted copy keyring-file to destination

```bash
# receive copy
# -x extract
# -C indicates the directory where to stream data
nc -l -p 9999 | pigz -c -d |  xbstream -x -C /mysql

# copy keyring
scp $IP_ORIGIN:/var/lib/mysql-keyring/keyring-encrypted /var/lib/mysql-keyring
chown -R mysql:mysql /var/lib/mysql-keyring

# consist basedir
xtrabackup --keyring-file-data=/var/lib/mysql-keyring/keyring-encrypted --prepare --target-dir=/mysql
chown -R mysql:mysql /mysql
```

# Physical copy

Note: need to stop both mysqls (origin and destination)

```bash
# source
tar -cf - /mysql | pigz -c | nc -w 10 -l -p 9999

# Destination
nc <ip_address> 9999 | pigz -d | tar -xf -

# if sources end , space bar to finish destination
```
