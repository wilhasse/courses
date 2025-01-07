# Testing

In MySQL create a encrypted table

```sql
CREATE TABLE `TESTE` (
  `ID` int NOT NULL,
  `TEXT` varchar(200) NOT NULL DEFAULT '',
  PRIMARY KEY (`ID`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 ENCRYPTION='Y'
```

Get keyring id  
You can see in the keyring file also:  
/var/lib/mysql-keyring/keyring-encrypted

```bash
mysql> SELECT * FROM performance_schema.keyring_keys;
+--------------------------------------------------+-----------+----------------+
| KEY_ID                                           | KEY_OWNER | BACKEND_KEY_ID |
+--------------------------------------------------+-----------+----------------+
| INNODBKey-fdd12291-78e9-11ed-ad12-506b8d77ad98-1 |           |                |
| INNODBKey-fdd12291-78e9-11ed-ad12-506b8d77ad98-2 |           |                |
+--------------------------------------------------+-----------+----------------+
2 rows in set (8,06 sec)
```

Copy ibd and run the decrypt
You need id and key

```bash
cd /data/percona-server/build/runtime_output_directory
./decrypt 1 fdd12291-78e9-11ed-ad12-506b8d77ad98 ~/keyring-encrypted ~/TEST.ibd ~/TEST_OK.ibd
```

Inspect if worked using  
https://github.com/twindb/undrop-for-innodb

Save table definition in a file: test.sql

```bash
# save 
cd undrop-for-innodb
./c_parser -6 -f TEST.ibd -t test.sql -s

# inspect result in output.txt
```

# Development

Percona Server Fork:  
https://github.com/wilhasse/percona-server

Build only this

```bash
cd build
cmake --build . --target decrypt
```

Visual Code: launch.json

```json
{
    "version": "0.2.0",
    "configurations": [        
        {
            "name": "Debug Percona MySQL",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/runtime_output_directory/decrypt",
            "args": [
                "1",
                "fdd12291-78e9-11ed-ad12-506b8d77ad98",
                "~/keyring-encrypted",
                "~/VERSAO.ibd",
                "~/VERSAO_u.ibd"
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "preLaunchTask": "Build decompress",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Load .gdbinit",
                    "text": "source ${workspaceFolder}/.gdbinit",
                    "ignoreFailures": true
                }
            ],
            "sourceFileMap": {
                "/data/percona-server": "${workspaceFolder}"
            },
            "logging": {
                "engineLogging": true,
                "trace": true,
                "traceResponse": true,
                "moduleLoad": true
            },
            "symbolLoadInfo": {
                "loadAll": true,
                "exceptionBreakpoint": "thrown"
            }
        },
        {
            "name": "C/C++: g++ build and debug active file",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly Flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "C/C++: g++ build active file",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

task.json

```json
{
    "version": "2.0.0",
    "tasks": [
      {
        "label": "Build decompress",
        "type": "shell",
        "command": "cmake",
        "args": [
          "--build",
          "${workspaceFolder}/build",   // The build dir
          "--target",
          "decrypt"                  // The target name
        ],
        "group": {
          "kind": "build",
          "isDefault": true
        },
        "problemMatcher": [
          // Or pick the correct problem matcher, e.g. "$gcc"
          "$gcc"
        ]
      }
    ]
  }
```
