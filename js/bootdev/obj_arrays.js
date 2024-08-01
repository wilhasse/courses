const getCountsByTitle = (movies) => {
  // ?

  const reviews = []
  for (const movie of movies) {

    if (!reviews[movie]) {
      reviews[movie] = 1
    } else {
      reviews[movie] += 1
    }
  }

  return reviews
}

// don't touch below this line

function test(movies) {
  const counts = getCountsByTitle(movies)
  for (const [ movie, count ] of Object.entries(counts)) {
    console.log(`'${movie}' has ${count} reviews`)
  }
  console.log('---')
}

test([
  'Ice Age',
  'The Forgotten',
  'The Northman',
  'American Psycho',
  'Ice Age',
  'Ice Age',
  'American Psycho'
])

test([
  'Big Daddy',
  'Big Daddy',
  'The Big Short',
  'The Big Short',
  'The Big Short'
])
