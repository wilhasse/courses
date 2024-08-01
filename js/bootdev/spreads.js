const uploadNewMovies = (oldMovies, newMovies) => {
  // ?
  return [...oldMovies,...newMovies]
}

// don't touch below this line

const oldMovies = [ 'Once Upon a Time', 'Django Unchained', 'The Hateful 8' ]
const newMovies = [ 'Inglorious Basterds', 'Dune' ]
const allMovies = uploadNewMovies(oldMovies, newMovies)

console.log('All movies database:')
logArray(allMovies)


function logArray(arr) {
  for (const a of arr) {
    console.log(` - ${a}`)
  }
  console.log('---')
}
