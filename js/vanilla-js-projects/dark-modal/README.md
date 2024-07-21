# Course

Vanilla Javascript Projects \
Classes & Browser APIs

A FrontendMasters course by Anjana Vakil

## Link

https://anjana.dev/vanilla-js-projects/

## Function / Scope and Arrow function

Arrow doesn't work here \
MDN: "Arrow functions don't have their own bindings to tuis, arguments or suer, and shoud not be used as methods"

```javascript
form.element.addEventListener("submit", (event) => {
  event.preventDefault();
  this.dialog.close();
});
```

Function this also doesn't work because of the scope

```javascript
form.element.addEventListener("submit", function (event) {
  event.preventDefault();
  this.dialog.close();
});
```

We need to use closure (scope)

```javascript
const dialog = this.dialog;
form.element.addEventListener("submit", function (event) {
  event.preventDefault();
  dialog.close();
});
```
