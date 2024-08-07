## Install

```bash
# uncompress tar
tar xf zig-linux-x86_64-0.12.0.tar.xz
# add bin to path
echo 'export PATH="$HOME/zig-linux-x86_64-0.12.0:$PATH"' >> ~/.bashrc
# verify
$ zig version
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
