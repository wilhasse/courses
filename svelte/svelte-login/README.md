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

layout.server.js - return data welcome_message: "welcome back"  
page.js - capture parent data from layout  
page.svelte - display parent data  

## Login process

routes/login/page.svelte - form page where user input user and password
routes/login/page.server.js - capture form data and check if user and password is ok, if not redirect to the root page again  
protected/page.svelte - protected page redirected after sucessfull login  
hooks.server.js - redirect to login page if not authenticated  

## Who is logged

hoook.server.js - set into event.locals the user got from the cookie.
layout.server.js - return the user in events.local in the load function
protected/page.svelte - get user from the load function and display it.

## Logout process

protected/page.svelte - there is a button that trigger an action to logout invoking signout url.  
hooks.server.js - in hooks I process the url if has signout , remove the username in the cookie.
