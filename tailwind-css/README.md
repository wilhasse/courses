# Course

FrontendMasters - Tailwind CSS  
Steve Kinney  

## Link

https://tailwind-workshop.vercel.app/introduction


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


