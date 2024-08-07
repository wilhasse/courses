# Course

Link: https://www.vimified.com
Learn Vim

# Lessons

## Chapter 1 Survival Essentials

### Vertical Movement

j - down  
k - up  
x - delete caracter

### Horizontal Movement

h - left  
l - right

### Introduction to Modes

i - insert mode  
ESC - normal mode

### Text Objects in Vim

w - next word  
shift + w - ignore punctuation  
b - previous word  
shift + b - ignore punctuation

### Word Based Movement

e - end of the word  
shift + e - treat punctuation as the same word

## Chapter 2 Commands Programmers Love

### Yanking and Putting

yy - copy line
p - paste

### Deletion

dd - delete line

### Visual Charracter Mode

v - start selection of text
navigate to the end selection (e, w, h, l, etc)
y - yank text

### Visual Line Mode

shift + v - start selection of text
navigate to the end selection (j,k)
y - yank text
p - paste text

# Others

## Keyboard

## Copy text inside vim

Copy line: yy  
Paste: p

## Copy text outside vim

Copy external text  
Insert Mode  
+p

# Undo / Redo

Undo: u
Redo: ctrl + r

# Remove line

dd

# Search string

In normal Mode
search forward: / + string + enter
search backward: ? + string + enter
next occurence: n
previous occurence: N

# Quit

without saving no changes: q
without saving discarding changes: :q!
save :wq
