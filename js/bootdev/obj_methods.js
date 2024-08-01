const user = {
  getFirstReview() {
    // ?
    return this.reviews[0]
  },
  reviews: [ 'I hate Ice Age', 'I didn\'t enjoy it at all', 'What a fabulous film' ],
  name: 'Bob Doogle'
}

// don't touch below this line
console.log(`${user.name}'s first review is: ${user.getFirstReview()}`)

const tree = {
  height: 256,
  color: 'green',
  cut() {
    this.height /= 2
  }
}

tree.cut()
console.log(tree.height)
// prints 128
