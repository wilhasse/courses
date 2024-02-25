# Course

Link: https://revealjs.com/course  

Mastering reveal.js   
Hakim El Hattab (creator)

# Videos

## Installation

```bsah
git clone https://github.com/hakimel/reveal.js.git
cd reveal.js
npm install
npm start
```
## Creating Slides

Markup
Slide Visibility
Links

Interesting feature data visibility can be hidden (don't show the slide just keep in your code)  
It also can be unconted which means that it can be shown as an bonus content but it doesn't count in the progress bar  

https://revealjs.com/slide-visibility/

## Configuration

Three ways to configure:

* Change code in Reveal.initialize ({ ... })  

* During the presentation typing in console: Reveal.configure({ autoSlide: 5000 });  

* Changing directly on the URL adding parameters:  
http://localhost:8000/?autoSlide=4000&loop=true