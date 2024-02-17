const canvas = document.querySelector('canvas');
const context = canvas.getContext('2d');

context.fillStyle = "yellow";
context.fillRect(0,0,100,100);

context.fillStyle = "black";
context.font = "50px sans-serif";
context.fillText("JS", 40, 90, 50); // (x, y, width, height

context.drawImage(video,0,0)