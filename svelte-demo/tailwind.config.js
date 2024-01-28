/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}',
            './node_modules/stwui/**/*.{svelte,js,ts,html}',
            './node_modules/tw-elements/dist/js/**/*.js'],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui"),
            require('stwui/plugin'),
            require("tw-elements/dist/plugin.cjs")]
  
}