# Files

- [Build Debian 11](./build_d11.md)
- [Build Debian 12](./build.md)
- [Environment](./environment.md)
- [Using gdb command line](./gdb_cmd.md)
- [Visual Code Config](./visual_code.md)
- [Update fork from upstream](./update_fork.md)
- [Logs](./logs.md)

# Utils

Help files to set up and debug mysql in Debian 12

- initialize a new data directory
- basic config (my.cnf) and grant all
- run mysqld in compiled version in percona-server

Reset data dir

```bash
./initialize_mysql.sh
```
Run / Shutdown

```bash
# inside screen (it will block terminal)
./run_mysql.sh
# shutdown mysql
```
Add root grants 

```bash
grant_mysql.sh
```