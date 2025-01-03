# Project

**Innodb Space**  
https://github.com/baotiao/inno_space  

My fork:  
https://github.com/wilhasse/inno_space

[Code detail](./innodb_space_doc.md)

## Usage

Generate sdi json file for the table

```bash
ibd2sdi /mysql/test/test.ibd > test.json
```

Print data

```bash
./inno -f /mysql/test/test.ibd -c dump-all-records -s test.json
```

Run with debug

Compile with debug symbols:

```bash
# ensure '-g' is present in Makefile
# CXXFLAGS = -Wall -W -DNDEBUG -g -O2 -std=c++11
make
```

Run:

```bash
gdb ./inno
```

Inside GDB, run:

```bash
gdb
(gdb) run -f ~/ACESSO_c.ibd -c dump-all-records -s ACESSO.json
```

## Problems

### 1) The original code didn't work with tables with composite primary key.  
OpenAi o1-pro was able to solve it in two shots.  

[Correction Explained](./innodb_space_o1_fix.md)

Correction here:  
https://github.com/wilhasse/inno_space/commit/ac844d248e95ee6c1427595ea008f73c8646fec4



