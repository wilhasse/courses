const leaderboard = await fetchLeaderBoard()
console.log(leaderboard)

// don't touch below this line

async function fetchLeaderBoard() {
  try {
    const response = await fetch('https://fantasyquest.servers')
    return response.json()    
  } catch (error) {
    return "Our servers are down, but we will be up and running soon"    
  }
}
