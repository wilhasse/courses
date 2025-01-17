# Install

Percona package  
https://repo.percona.com/pdps-8.0/apt/pool/main/j/jemalloc/

```bash
apt-get install libjemalloc1
```

# Percona

Enable jemalloc1

```bash
locate libjemalloc.so.1
#/usr/lib/x86_64-linux-gnu/libjemalloc.so.1

pico /etc/default/mysql
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.1

# Restart MySQL service
sudo systemctl daemon-reload
sudo systemctl restart mysql
```

Verifying

```bash
# lsof loaded libraries
root@CE00DB02:/home/cslog# sudo lsof -p $(pidof mysqld) | grep jemalloc
mysqld  120016 mysql  mem       REG              254,0     219848    274778 /usr/lib/x86_64-linux-gnu/libjemalloc.so.1

# runtime libraries
root@CE00DB02:/home/cslog# sudo ls -l /proc/$(pidof mysqld)/map_files/ | grep jemalloc
lr-------- 1 mysql mysql 64 jan 16 22:37 7f8d07660000-7f8d07663000 -> /usr/lib/x86_64-linux-gnu/libjemalloc.so.1
lr-------- 1 mysql mysql 64 jan 16 22:37 7f8d07663000-7f8d0768c000 -> /usr/lib/x86_64-linux-gnu/libjemalloc.so.1
lr-------- 1 mysql mysql 64 jan 16 22:37 7f8d0768c000-7f8d07694000 -> /usr/lib/x86_64-linux-gnu/libjemalloc.so.1
lr-------- 1 mysql mysql 64 jan 16 22:37 7f8d07694000-7f8d07696000 -> /usr/lib/x86_64-linux-gnu/libjemalloc.so.1
lr-------- 1 mysql mysql 64 jan 16 22:37 7f8d07696000-7f8d07697000 -> /usr/lib/x86_64-linux-gnu/libjemalloc.so.1
```

# Hugepages

For MySQL with jemalloc, it's generally recommended to disable transparent huge pages (THP) because:

- MySQL's memory access patterns don't benefit much from huge pages
- THP can cause performance issues and memory fragmentation
- jemalloc works better with standard pages

```bash
### Check current status
cat /sys/kernel/mm/transparent_hugepage/enabled

### Temporarily disable
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag
```

## Make it permanent

Create /etc/systemd/system/

```bash
[Unit]
Description=Disable Transparent Huge Pages
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c "echo 'never' > /sys/kernel/mm/transparent_hugepage/enabled && echo 'never' > /sys/kernel/mm/transparent_hugepage/defrag"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable disable-thp
sudo systemctl start disable-thp
```

Verifying

```bash
# Check status
sudo systemctl status disable-thp

# Verify THP is disabled
cat /sys/kernel/mm/transparent_hugepage/enabled
```