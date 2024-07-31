const movieTitle = null


//Null values are "falsy"
//"Falsy" means that a value evaluates to false when cast to a boolean. Here are some examples of "falsy" values:
//
//false (false boolean)
//0 (number zero)
//'' (empty string)

// The 'not' operator implicitly
// casts movieTitle to a boolean value
const movieHasNoTitle = !movieTitle

console.log(`The movie does not have a title: ${movieHasNoTitle}`)
