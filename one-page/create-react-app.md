# Install

## Nodejs

Specific version

```bash
nvm install 22.14.0
nvm use 22.14.0
```

## React App

```bash
# Create a new Vite project with React
npm create vite@latest mysql-delay-analyzer -- --template react

# Navigate to the project directory
cd mysql-delay-analyzer

# Install Tailwind CSS and its dependencies
npm install -D tailwindcss postcss autoprefixer

# Run
npm install
npm run dev
```

## Plataform Error (Windows)

Error:

```bash
D:\svn\ticslog_trunk\docker\csmysqldelay\mysql-delay-analyzer\node_modules\rollup\dist\native.js:64
                throw new Error(
                      ^
Error: Cannot find module @rollup/rollup-win32-x64-msvc. npm has a bug related to
```

Fix:

```bash
npm install @rollup/rollup-win32-x64-msvc
npm run dev
```

## Tailwind

Edit the tailwind.config.js

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

src/index.css:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```
