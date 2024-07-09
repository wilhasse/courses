# Material

Zig Guide
https://zig.guide/

## Install

```bash
# uncompress tar
tar xf zig-linux-x86_64-0.12.0.tar.xz
# add bin to path
echo 'export PATH="$HOME/zig-linux-x86_64-0.12.0:$PATH"' >> ~/.bashrc
# verify
$ zig version
```

## Hello

Hello World
Run Test

```bash
cd hello
zig run main.zig
zig test test_pass.zig
zig test test_fail.zig
```
