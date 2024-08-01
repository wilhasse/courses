// ?

// don't touch below this line

function logArray(arr) {
  console.log('logging array...')
  for (const a of arr) {
    console.log(` - ${a}`)
  }
  console.log('---')
}

const movie = []
movie.push('the dark knight')
logArray(movie)
movie.push('the notebook')
logArray(movie)
console.log(movie[0])
