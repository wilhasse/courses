## Video 1 - Signals

https://www.youtube.com/watch?v=PC4Jx6ZBfTg  
Svelte 5 Rune Reactivity Explained: Understanding Runtime Reactivity vs Compile-time Reactivity  

Files:
- svelte4.js
- signal.js
- svelte5.js

## Video 2 - Basic

https://www.youtube.com/watch?v=-SM77ksjpJI  
Svelte 5 Runes Demystified (1/4) - Signal Reactivity Basics

https://svelte.dev/blog/runes  
Introduction to Runes

![Stores vs Signals](images/video2a.png)

Stelve 4

```html
<script>
let name = 'world'
let caractersInName;
$: {
    caractersInName = name.length
}
</script>

<input type="text" bind:value={name} />
<h1>Hello {name}!</h1>
{caractersInName}
```

Stelve 4 - function 
name is still binded 
caracter count doesn't work

```html
<script>
let name = 'world'
let caractersInName;
function countLetters() {
    caractersInName = name.length
}
$: {
    countLetters()
}
</script>

<input type="text" bind:value={name} />
<h1>Hello {name}!</h1>
{caractersInName}
```

Stelve 5

It's working now

```html
<script>
let name = $state('world')
let caractersInName = $state(0);
function countLetters() {
    caractersInName = name.length
}

$effect((): {
    caractersInName = name.length
})
</script>

<input type="text" bind:value={name} />
<h1>Hello {name}!</h1>
{caractersInName}
```

![In Deph](images/video2b.png)
