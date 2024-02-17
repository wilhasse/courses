import { say } from 'cowsay'

const currentDir = process.cwd();
const message = 'Moo are here:\n' + currentDir
console.log(say({ text: message}));


