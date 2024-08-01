const isSuperReview = (authorAccountAge, authorReputation) => {
  if (authorAccountAge > 2) {
    return false
  }
  return authorReputation > 100
}

// don't touch below this line

function isFunctionType(f) {
  // eslint-disable-next-line no-prototype-builtins
  return f.hasOwnProperty('prototype')
}

if (isFunctionType(isSuperReview)) {
  console.log('isSuperReview is a classic function')
} else {
  console.log('isSuperReview is a fat arrow function')
}

const isSuper = isSuperReview(50, 200)
console.log(`The review is super: ${isSuper}`)

const isSuper2 = isSuperReview(1, 200)
console.log(`The review is super: ${isSuper2}`)
