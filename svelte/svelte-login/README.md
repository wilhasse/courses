# Project

Jannik Wempe
Authentication in Svelte using cookies
https://blog.logrocket.com/authentication-svelte-using-cookies/

## Create

```bash
# create a new project 
npm create svelte@latest svelte-login
npm install
npm run dev
```

Select Skeleton project 
No Typescript

## Tailwind CSS

```bash
npx svelte-add@latest tailwindcss
npm install
```

## Loading parent data

layout.server.js - return data welcome_message: "welcome back",
page.js - capture parent data from layout
page.svelte - display parent data
