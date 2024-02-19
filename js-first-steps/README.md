# Course

First Steps to Professional

A FrontendMasters course by Anjana Vakil

https://anjana.dev/javascript-first-steps/

## Introduction

![Grand Plan](images/grand_plan.png)

## DOM

Exercise: Tic Tac Toe  
Type commands in the console to retrieve:  
1. all the p elements  
2. the text "X"  
3. the number of squares in the board  
4. the text "A game you know"

```javascript
document.getElementsByTagName("p") //1
document.querySelector('#p1-symbol').textContent //2
document.querySelectorAll('.square').length //3
document.querySelector('h2').textContent//4
```

Exercise: Changing the Board  
1. Change the player names to you & neighbor  
2. Swap the player symbols  
3. Change subtitle to "A game you know and love"  

```javascript
document.querySelector('#p1-name').textContent = 'Willian'
document.querySelector('#p2-name').textContent = 'Arthur'
document.querySelector('#p1-symbol').textContent = 'O'
document.querySelector('#p2-symbol').textContent = 'X'
document.querySelector('header h2').append(' and love')
```
