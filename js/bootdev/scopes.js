function isMovieValid(title) {
  
  function movieLength(title) {
    return len
  }

  // len could be here or before function movieLength
  // it also works afterward
  const len = title.length

  movieLength(title)
  return len > 2
}

function test(title) {
  const valid = isMovieValid(title)
  console.log(`'${title}' is valid: ${valid}`)
}

test('The League of Extraordinary Gentlemen')
test('Hunt for Red October')
test('007')
test('')
test('12')
