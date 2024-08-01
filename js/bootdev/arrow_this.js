const author = {
  firstName: 'Lane',
  lastName: 'Wagner',
  getName() {
    return `${this.firstName} ${this.lastName}`
  }
}
console.log(author.getName())
// Prints: Lane Wagner

const author2 = {
  firstName: 'Lane',
  lastName: 'Wagner',
  getName: () => {
    return `${this.firstName} ${this.lastName}`
  }
}
console.log(author2.getName())
// Prints: undefined undefined
// because the parent scope (the scope outside of the author object)
// never defined .firstName and .lastName properties
