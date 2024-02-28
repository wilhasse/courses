# Course

Link: https://observablehq.com/collection/@anjana/functional-javascript-first-steps  

## What is Functional Programming

Pure vs Impure Functions  
Why Functional JavaScript  
Side Effects  
Pure Functions Exercise  
Pure Functions Solution

https://codewords.recurse.com/issues/one/an-introduction-to-functional-programming


This is not a pure function. Why?

```javascript
getDate = ƒ()

function getDate() {
  return new Date().toDateString();
}
```

## Staying out of the Loop with Recursion

Recursion  
Iteration vs Recursion Exercise  
Iteration vs Recursion Solution  
Recursion Performance & Proper Tail Calls


## Recursion

Interative (classical loop)

```javascript
function sum (numbers) {
  let total = 0;
  for (i = 0; i < numbers.length; i++) {
    total += numbers[i];
  }
  return total;
}

sum([0,1,2,3,4]); // 10
```

Recursive (function call itself)

```javascript
function sum (numbers) {
  if (numbers.length === 1) {
    // base case
    return numbers[0];
  } else {
    // recursive case
    return numbers[0] + sum(numbers.slice(1));
  }
}

sum([0,1,2,3,4]); // 10
```

## Higher order functions

First-class functions  
can be passed around as values  
(like callbacks!)

Higher-order functions  
take other functions  
as input/output

## Currying

```javascript
function greet(greeting, name) {
  return `${greeting}, ${name}!`;
}

function curryGreet(greeting) {
  return function (name) {
    return `${greeting}, ${name}!`;
  }
};

const greetItal = curryGreet("Ciao");
greetItal("Alonzo"); // "Ciao, Alonzo!"

const greetTex = curryGreet("Howdy");
greetTex("Alonzo"); // "Howdy, Alonzo!"
greetTex("Alan"); // "Howdy, Alan!"
```

## Composing functions

```javascript
var ender = (ending) => (input) => input + ending;

var adore = ender(" rocks");
var announce = ender(", y'all");
var exclaim = ender("!");

var hypeUp = (x) => exclaim(announce(adore(x)));
hypeUp("JS"); // "JS rocks, y'all!"
hypeUp("FP"); // "FP rocks, y'all!"
```

## Immutability

```javascript
const oldCities = ["Delhi", "Bombay", "Bangalore"];

const newCities = oldCities.map((city) => {
  if (city === "Bombay") return "Mumbai";
  if (city === "Bangalore") return "Bengaluru";
  return city;
});

newCities; //["Delhi", "Mumbai", "Bengaluru"]
oldCities; //["Delhi", "Bombay", "Bangalore"]
```
