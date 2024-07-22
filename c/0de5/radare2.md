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
