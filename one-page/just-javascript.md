# Course

Link: https://justjavascript.com  

Explore the JavaScript Universe  
Dan Abramov (co-author of Redux and Create React App)  
Maggie Appleton (Illustrator)  

# Modules

## Mental Models

This code below has an accidental bug which changes the value of the original tittle.  
Why?

```javascript
function duplicateSpreadsheet(original) {
  if (original.hasPendingChanges) {
    throw new Error('You need to save the file before you can duplicate it.');
  }
  let copy = {
    created: Date.now(),
    author: original.author,
    cells: original.cells,
    metadata: original.metadata,
  };
  copy.metadata.title = 'Copy of ' + original.metadata.title;
  return copy;
}
```

## The JavaScript Universe

A value is a fundamental concept in JavaScript—so we can’t define it through other terms.  
Instead, we’ll define it through examples. Numbers and strings are values. Objects and functions are values, too.  
In our JavaScript universe, values float in space. Why ?  It is still not possible to answer that question.    

There are two types of values:
* Primitive values: numbers, booleans, strings, etc
* Objects and Functions

typeof

```javascript
console.log(typeof(2)); // "number"
console.log(typeof("hello")); // "string"
console.log(typeof(undefined)); // "undefined"
```


## Values and Variables

## Studying from the Inside

## Meeting the Primitive Values

## Meeting Objects and Functions

## Equality of Values

## Properties

## Mutation

## Prototypes