# Configuring visual code to debug

How to configure VS code to be able to debug MySQL C/C++ code

# launch.json

```json
{
    "version": "0.2.0",
    "configurations": [        
        {
            "name": "Debug Percona MySQL",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/runtime_output_directory/mysqld",
            "args": [
                "--defaults-file=/data/my.cnf",
                "--datadir=/data/mysql",
                "--log-error=/data/mysql_error.log",
                "--debug=d:t:i:o,/data/mysqld.trace"
            ],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
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
            "preLaunchTask": "Build decompress"
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

# tasks.json

Execute cmake to build the project before debugging

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
        "decompress"                  // The target name
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

# c_cpp_properties.json

To be able to find includes, generate file compile_commands.json  
It only worked by running cmake in the build repo

```bash
cd /data/percona-server/build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

cslog@mysql-8-src:/data/percona-server/build$ ls -la compile_commands.json 
-rw-r--r-- 1 cslog cslog 7354036 jan  1 22:47 compile_commands.json
```

```json
{
    "configurations": [
        {
            "name": "Linux",
            "compileCommands": "${workspaceFolder}/build/compile_commands.json",
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

I have problem with include, I wasn't able to debug because IntelliSense didn't recognize my include paths in the project.  

I troubleshooted inspecting compile_commands.json, you can see config running diagnostics:

Run “C/C++: Log Diagnostics” in VS Code
Open the command palette (Ctrl+Shift+P or Cmd+Shift+P) and run:

```bash
C/C++: Log Diagnostics
```

