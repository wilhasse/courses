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

# Text Editor

Sublime install

```bash
sudo apt-get install sublime
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/sublimehq-archive.gpg > /dev/null
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
sudo apt-get update
sudo apt-get install sublime-text
```

I export my display to Windows using X410
Windows IP is 10.1.1.139

```bash
export DISPLAY=10.1.1.139:0.0
subl mysql_error.log
```

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