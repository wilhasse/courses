// avoid var
// in this cade x is still accesible outside if scope
function printX(shouldSet) {
  if (shouldSet) {
    var x = 2
  }
  console.log(x);
  // Prints: 2
}
printX(true)

function printY(shouldSet) {
  if (shouldSet) {
    let x = 2
  }
  // ReferenceError: x is not defined
  // error if discomment the line below
  // console.log(x);
}
printY(true)
