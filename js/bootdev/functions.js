function getLabel(numStars) {
  // ?
  if (numStars >=8) {
    return "great"
  } 

  if (numStars >=4) {
    return "okay"
  } 

  if (numStars <=3) {
    return "awful"
  } 
}

// don't touch below this line

function test(numStars) {
  console.log(`Stars=${numStars}, The movie was ${getLabel(numStars)}`)
}

test(10)
test(9)
test(8)
test(7)
test(6)
test(5)
test(4)
test(3)
test(2)
test(1)
test(0)
