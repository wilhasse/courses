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

Primitive Values Are Immutable

```javascript
let reaction = 'yikes';
reaction[0] = 'l';
console.log(reaction);
//yikes
```

Arrays are not primitives so I can change it

```javascript
let arr = ['y','i','k','e','s'];
//undefined
arr[0] = 'l';
//'l'
arr
//['l', 'i', 'k', 'e', 's']
```

But variables are wire, I can change it despite not values

```javascript
let pet = 'Narwhal';
pet = 'The Kraken';
console.log(pet);
// The Kraken
```
Summarizing:  
Variables are not values.  
Variables point to values.  

## Studying from the Inside

The foundation of our mental model is values.  
Each value belongs to a type.  
Primitive values are immutable.  
We can point to values using “wires” we call variables.

## Meeting the Primitive Values

![Values](images/celestialspheres-v2.png)

```javascript
console.log(typeof(undefined)); 
// "undefined"
```

Strings Aren’t Objects

```javascript
console.log(typeof('Hello'));
//string
```

Numbers

```javascript
let scale = 0;
let a = 1 / scale;
// Infinity
let b = 0 / scale;
// NaN
let c = -a;
// -Infinity
let d = 1 / c;
// -0

console.log(typeof(NaN)); 
// "number"
```

## Meeting Objects and Functions

Do Objects Disapper? No.

```javascript
let shrek = {};
let donkey = {};
//undefined
shrek = null;
//null
shrek
//null
```

How many different values does this code pass to console.log? 7

```javascript
for (let i = 0; i < 7; i++) {
  console.log(function() {});
}
//7 f() {}
```

## Equality of Values

```javascript
console.log(2 === 2); 
// true
console.log({} === {}); 
// false
Object.is(2,2);
//true
Object.is({},{});
//false
```

```javascript
console.log(NaN === NaN);
//false
Object.is(NaN,NaN);
//true
```

```javascript
let width = 0;
let height = -width;
console.log(width === height); 
// true
console.log(Object.is(width, height)); 
// false
```

Loose Equality ==  
Don't use it !

```javascript
console.log([[]] == ''); 
// true
console.log(true == [1]); 
// true
console.log(false == [0]); 
// true
```

## Properties

```javascript
let sherlock = {
  surname: 'Holmes',
  address: { city: 'London' }
};
let john = {
  surname: 'Watson',
  address: sherlock.address
};
john.surname = 'Lennon';
john.address.city = 'Malibu';

console.log(sherlock.surname); 
// Holmes
console.log(sherlock.address.city); 
// Malibu ??? 
console.log(john.surname); 
// Lennon
console.log(john.address.city); 
// Malibu
```

The sherlock.address.city is Maliby because john.address is pointing to sherlock.address.   
When john.address has changed to Malibu it already changed sherlock.address, the variable it was point to.  
I fresh new shelock object keep London in this case:


```javascript
let sherlock2 = {
  surname: 'Holmes',
  address: { city: 'London' }
};
sherlock2.address.city
//'London'
```

## Mutation

Not mutating properties

```javascript
let sherlock = {
  surname: 'Holmes',
  address: { city: 'London' }
};
let john = {
  surname: 'Watson',
  address: { city: 'London' }
};
john.surname = 'Lennon';
john.address.city = 'Malibu';

console.log(sherlock.surname); 
// Holmes
console.log(sherlock.address.city); 
// London !! 
console.log(john.surname); 
// Lennon
console.log(john.address.city); 
// Malibu
```

## Prototypes

```javascript
let human = {
  teeth: 32
};

let gwen = {
  age: 19
};

console.log(gwen.teeth);
//undefined

let human = {
  teeth: 32
};

let gwen = {
  // We added this line:
  __proto__: human,
  age: 19
};

console.log(gwen.teeth);
//32
```

```javascript
let human = {
  teeth: 32
};

let gwen = {
  __proto__: human,
  // This object has its own teeth property:
  teeth: 31
};
console.log(human.teeth); 
// 32
console.log(gwen.teeth); 
// 31
```

```javascript
let human = {
  teeth: 32
};

let gwen = {
  __proto__: human,
  // Note: no own teeth property
};

gwen.teeth = 31;

console.log(human.teeth); 
// 32
console.log(gwen.teeth); 
// 31
```
