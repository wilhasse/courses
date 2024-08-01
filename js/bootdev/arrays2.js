function getCleanRank(reviewWords) {
  // ?
  let bad = 0
  for (const word of reviewWords)  {

    if (word === 'dang' || word === 'shoot' || word === 'heck') {
      bad += 1      
    }
  }

  if (bad === 0) {
    return 'clean'
  }

  if (bad === 1) {
    return 'dirty'
  }

  if (bad  >= 2) {
    return 'filthy'
  }
}


// Don't edit below this line

function test(reviewWords) {
  const cleanRank = getCleanRank(reviewWords)
  console.log(`'${reviewWords}' has rank: ${cleanRank}`)
}

test([ 'avril', 'lavigne', 'has', 'best', 'dang', 'tour' ])
test([ 'what', 'a', 'bad', 'film' ])
test([ 'oh', 'my', 'heck', 'I', 'hated', 'it' ])
test([ 'ripoff' ])
test([ 'That', 'was', 'a', 'pleasure' ])
test([ 'shoot!', 'I', 'cant', 'say', 'I', 'liked', 'the', 'dang', 'thing' ])
test([ 'shoot', 'dang', 'heck' ])
