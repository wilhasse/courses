function getDomainNameFromURL(url) {
  // ?
  return new URL(url).hostname
}

// don't touch below this line

const bootdevURL = 'https://boot.dev/courses/learn-python'
const domainName = getDomainNameFromURL(bootdevURL)
console.log(`The domain name for ${bootdevURL} is ${domainName}`)
