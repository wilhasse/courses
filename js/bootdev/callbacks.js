function onReview(reviewText, callbackFunction) {
  console.log(`Review: ${reviewText}`);
  // ?
  callbackFunction(reviewText)
}

function main() {
  const ohBrotherWhereArtThouReview = 'Stellar movie!'
  const twentyTwelveIceAgeReview = 'Not my favorite'
  // ?
  onReview(ohBrotherWhereArtThouReview,saveToDatabase)
  onReview(twentyTwelveIceAgeReview,saveToDatabase)
}

// Don't edit below this line

function saveToDatabase(reviewText) {
  console.log(`Saving '${reviewText}' to database...`)
}

main()
