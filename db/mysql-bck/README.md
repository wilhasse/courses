# Introduction

Percona Xtrabckup Full nd Incremental  
It uses ssb in courses/db/ssb

- Perform a full backup
- Perform a increental backup
- Remove datadir
- Restore incremental and full backup

# Preparation

Install Debian 12 Percona xtrabackup

```bash
curl -O https://repo.percona.com/apt/percona-release_latest.generic_all.deb
sudo apt install gnupg2 lsb-release ./percona-release_latest.generic_all.deb
sudo apt update
percona-release setup pxb-80
sudo apt install percona-xtrabackup-80
```

Run ssb to create tables and load data

```bash
# populate ssb data
cd courses/db/ssb  
./create_data.sh localhost mysql
./load_data.sh localhost test /data/100M/

# query rows
mysql ssb -u root -p
mysql> SELECT COUNT(*) FROM lineorder;
+----------+
| COUNT(*) |
+----------+
|   600597 |
+----------+
1 row in set (0,02 sec)
```

# Full backup

```bash
# test dir
mkdir xtra-test
cd xtra-test
sudo xtrabackup --backup --stream=xbstream --extra-lsndir=meta -u backup 2> log_full.txt | pigz -c > backup.gz
# ls backup amd meta dir
drwxr-xr-x 4 cslog cslog     4096 nov  9 10:32 .
drwx------ 7 cslog cslog     4096 nov  9 10:30 ..
-rw-r--r-- 1 cslog cslog 87685501 nov  9 10:32 backup.gz
-rw-r--r-- 1 cslog cslog    44556 nov  9 10:32 log_full.txt
drwxr-x--- 2 root  root      4096 nov  9 10:32 meta
```

# Incremental backup

Add more data

```bash
cd ~/courses/db/ssb
# don't forget 1 to append data
./load_data.sh localhost test /data/10M/ 1
mysql> SELECT COUNT(*) FROM lineorder;
+----------+
| COUNT(*) |
+----------+
|   617728 |
+----------+
1 row in set (0,03 sec)
```

Incremental backup

```bash
# get LSN
cd ~/xtra-test
cslog@dbgen:~/xtra-test$ sudo cat meta/xtrabackup_checkpoints | grep to_lsn | awk '{print $3}'
3434817512

# backup
sudo xtrabackup --backup --stream=xbstream --incremental-lsn=3434817512 --extra-lsndir=meta2 -u backup 2> log_inc.txt | pigz -c > backup_incremental.gz

# verify new backup
-rw-r--r-- 1 cslog cslog  84M nov  9 10:32 backup.gz
-rw-r--r-- 1 cslog cslog  39M nov  9 10:37 backup_incremental.gz
-rw-r--r-- 1 cslog cslog  44K nov  9 10:32 log_full.txt
-rw-r--r-- 1 cslog cslog  44K nov  9 10:37 log_inc.txt
drwxr-x--- 2 root  root  4,0K nov  9 10:32 meta
drwxr-x--- 2 root  root  4,0K nov  9 10:37 meta2
```

# Restore backup

Stop database remove datadir

```bash
sudo -s
systemctl stop mysql
rm -rf /var/lib/mysql
ls -la /var/lib/mysql
ls: não foi possível acessar '/var/lib/mysql': Arquivo ou diretório inexistente
```

Restore full backup

```bash
mkdir full
cd full
gunzip -c ../backup.gz | xbstream -x
xtrabackup --prepare --apply-log-only --target-dir=.
```

Restore incremental to full backup

```bash
# inc
mkdir inc
cd inc
gunzip -c ../backup_incremental.gz | xbstream -x
xtrabackup --prepare --apply-log-only --target-dir=../full --incremental-dir=.

# go back to full and apply log
cd ..
cd full/
xtrabackup --prepare --target-dir=.
```

Move back to mysql and test it

```bash
cd ..
mv full /var/lib/mysql
chown -R mysql:mysql /var/lib/mysql
systemctl start mysql
mysql ssb -u root -p
mysql> SELECT COUNT(*) FROM lineorder;
+----------+
| COUNT(*) |
+----------+
|   617728 |
+----------+
1 row in set (0,40 sec)
```
