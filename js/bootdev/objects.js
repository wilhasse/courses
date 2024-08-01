function getMovieRecord(title, stars, username) {
  // ?
  return {
    title: title,
    stars: stars,
    username: username,
    id: title + '-' + username
  }
}

// Don't touch below this line

logObject(getMovieRecord('oh brother where art thou', 3, 'wagslane'))
logObject(getMovieRecord('frozen', 5.5, 'elonmusk'))
logObject(getMovieRecord('toy story', 4, 'prince'))

function logObject(obj) {
  for (const key in obj) {
    console.log(` - ${key}: ${obj[key]}`)
  }
  console.log('---')
}
