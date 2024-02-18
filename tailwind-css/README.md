# Course

FrontendMasters - Tailwind CSS  
Steve Kinney  

## Link

https://tailwind-workshop.vercel.app/introduction

## Visual Studio

Install Extension Tailwind CSS IntelliSense

## Tailwind with Vite

```bash
npm init vite@latest
cd project
npm install
npm install autoprefixer
npm i -D tailwindcss@latest
npx tailwindcss init -p
```

add index.html or other pages to tailwind.config.js content section

```js
  content: [
    "./index.html"
  ],
```

replace styles.css to

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

index.html

add styles 

```html
    <link rel="stylesheet" href="/style.css" />
```

## Basic Styling

Styling a Button 

```html
bg-blue-500
text-white rounded
p-4 px-1 py-2
border-2 border-blue-700
```

Sizing and Spacing 
```html
h-72 w-96 bg-blue-200 
px-4 py-10">
mx-20 my-10
```

Text Sizing 
```html
text-xl text-2xl
```

Adding Spacing 

```html
space-y-4
space-w-4
```

Adding Dividers 

```html
divide-y-4
divide-x-4
```
## Variant

## Layout

## Plugin

## Useful Tricks

