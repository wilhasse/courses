# Project

**Innodb Space**  
https://github.com/baotiao/inno_space  

My fork:  
https://github.com/wilhasse/inno_space

- [Code detail by o1-pro](./innodb_space_doc.md)

## Usage

Generate sdi json file for the table

```bash
ibd2sdi /mysql/test/test.ibd > test.json
```

Print data

```bash
./inno -f /mysql/test/test.ibd -c dump-all-records -s test.json
```

## Problems

- The original code didn't work with tables with composite primary key.
Correction here:  
https://github.com/wilhasse/inno_space/commit/ac844d248e95ee6c1427595ea008f73c8646fec4

- It doesn't read all types VARCHAR (length may vary), unsigned int , etc
Need to implement custom logic

