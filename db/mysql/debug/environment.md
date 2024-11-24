# Remote Visual Code

Debuging from Windows on Linux

Genereate SSH Keys
In your homedir

```prompt
ssh-keygen -t rsa -b 4096 -C 10.1.1.148
type .ssh\id_rsa.pub | ssh 10.1.1.148 "cat >> .ssh/authorized_keys"
```

Extensions

- C/C++
- C/C++ Extension Pack
- C/C++ Themes
- CMake
- GitLens
- Hex Editor

# lauch.json

Arguments passing to mysql.

Usually 
- my.cnf
- where to log error
- debug what to trace

Example:

```json
    "args": [
    "--defaults-file=/data/my3306/conf/my.cnf",
    "--log-error=/data/my3306/mysql_error.log",
    "--debug=d:t:i:o,/data/my3306/mysqld.trace"
```