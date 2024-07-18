# Course

0de5

# Tools

https://github.com/radareorg/radare2

git clone https://github.com/radareorg/radare2
radare2/sys/install.sh

# Commands

```bash
# compile without opitmization
gcc -O0 -g -o welcome welcome.c

# run r2
r2 welcome
# analyze program
aaa
# list all functions (need to use aaa first)
afl
# Visual edit configuration
Ve
# go to the function
s sym.main
# print the assemble
pdi
# go and print the assemble
pdf pdf @dbg.make_it_higher
```
