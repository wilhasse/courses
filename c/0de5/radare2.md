# Tutorial

radare2
https://youtube.com/playlist?list=PLCwnLq3tOElpDyvObqsAI5crEO9DdBMNA&si=Rt9A2IE4MK08KQ3g

# Install

https://github.com/radareorg/radare2

git clone https://github.com/radareorg/radare2
radare2/sys/install.sh

# Interface Overview and Simple Analysis

```bash
# compile without opitmization
gcc -O0 -g -o r2_walkthrough r2_walkthrough.c

# run r2
r2 welcome
# analyze program
aaa
# list all functions (need to use aaa first)
afl
# go to main function
s main
# print assembly
pdf
# visual mode
V
# keyboard "p" to go to different views
p
# keyboard shift b enter in graph mode, space bar returns
shift+b
# keyboard shift colon enter command prompt down
shift+;
# another visual ollie debug
V!
# Visual edit configuration
Ve
```

# Strings, Cross References, Simple Assembly Analysis

```bash
# strings in the data section
iz
# search widestrings
izz~Login
# print 1@ follow vaddr of string
pd 1@0xHEX
# print all data
axt @ @ str.*
# go to location and prind assembly
s sym.result
pdf
# check when the function is called (cross reference)
axt
# see the function and go to it
s sym.check_password
# convert hex to text to see mov data
rax2 -s 0xHEX
```

# r2pm Ghidra Decompiler Usage in R2

```bash
# install hidra
sudo apt-get update
sudo apt-get install cmake
sudo apt-get install pkg-config
apt-get install meson
r2pm -ci r2ghidra

r2 a.out
aaa
s main
pdg

# install r2dec
git clone --depth=10 --recursive https://github.com/wargio/r2dec-js
cd r2dec-js/
meson setup build
sudo ninja -C build install

r2 a.out
aaa
s main
pdd
```

# Debugging Part 1 Runtime Analysis and Binary Patching

```bash
# run with analysis "aaa"
r2 -AA a.out
# list all colors
eco
# change to gentoo
eco gentoo
# change to display
V (p p)
# debug
shift + ;
# set breakpoint
db main
# re-open to debug
ood
# continue breakpoint (hit it)
dc
# change breakpoint
f2
# change jne (75) to je (74)
s 0x563e476601a7
pd 1
wx 74
pd 1
# go back to debug
f8
```

Patch binary

```bash
pdf @sym.check_password
s 0x000011a7
pd 1
wx 74
pd1
q
```
