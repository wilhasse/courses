## Install Zig

Zig

```bash
# uncompress tar
tar xf zig-linux-x86_64-0.12.0.tar.xz
# add bin to path
echo 'export PATH="$HOME/zig-linux-x86_64-0.12.0:$PATH"' >> ~/.bashrc
# verify
$ zig version
```

Zfs

```bash
git clone https://github.com/zigtools/zls
zig build
mv zig-out/bin/zls /usr/local/bin/
mkdir .config
git clone https://github.com/nvim-lua/kickstart.nvim nvim
```

## Debug in VSCode

Linux 

Install extensions
- Remote SSH (from Windows)
- CodeLLDB
- C/C++ Microsoft
- Zig Language (ziglang)

Create task  
Ctrl + Alt + P task  
<your-project-name>/.vscode/tasks.json  

```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "zig build",
            "type": "shell",
            "command": "zig",
            "args": [
                "build",
                "--summary",
                "all"
            ],
            "options": {
                "cwd": "${workspaceRoot}"
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            },
            "problemMatcher": [],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

Create launch.json  
Substitute <your-project-name> for your bin name  

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug",
            "program": "${workspaceFolder}/zig-out/bin/<your-project-name>",
            "args": [],
            "cwd": "${workspaceFolder}",
            "preLaunchTask": "zig build"
        },
    ]
}
```


## Install Neovim

```bash
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
sudo rm -rf /opt/nvim
sudo tar -C /opt -xzf nvim-linux64.tar.gz

# add .profile
set -o vi
alias vi=nvim
alias vim=nvim

# exit terminal log in again
exit
```

Custom script  
https://codeberg.org/dude_the_builder/zig_master/src/branch/master/01_neovim_setup/init.lua

```bash
mv init.lua .config/nvim/
vim .config/nvim/init.lua
```

## Configure vim

Install plugin manager

```bash
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

Add to ~/.vimrc

```bash
call plug#begin('~/.vim/plugged')
Plug 'ziglang/zig.vim'
call plug#end()
```

Reopen Vim and run the command:

```bash
vim
:PlugInstall
:q!
```

## Course Ziglings

Ziglings - 107 exercises
https://codeberg.org/ziglings/exercises/

Page My solution
[ziglings](./ziglings.md)

## Exercises

Trying the language
Random exercises

[zig-exercises](./zig-exercises/)

## RocksDB

Testing Zig with RocksDB

[rocksdb](./rockskmin/)
