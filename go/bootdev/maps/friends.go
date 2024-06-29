package friends

/*

A suggested friend is someone who is not a direct friend of the user but is a
direct friend of one or more of the user's direct friends. Each suggested friend
should appear only once in the slice, even if they are found through multiple direct friends.

friendships := map[string][]string{
    "Alice":   {"Bob", "Charlie"},
    "Bob":     {"Alice", "Charlie", "David"},
    "Charlie": {"Alice", "Bob", "David", "Eve"},
    "David":   {"Bob", "Charlie"},
    "Eve":     {"Charlie"},
}

suggestedFriends := findSuggestedFriends("Alice", friendships)
// suggestedFriends = [David, Eve]

*/

func hasElement(slice []string, target string) (bool) {
	for _, v := range slice {
		if v == target {
			return true
		}
	}
	return false
}

func findSuggestedFriends(username string, friendships map[string][]string) []string {

	// friends
	var friends []string
	var ofriends []string
	var mutualFriends []string

	friends = friendships[username]
	for _,directFriend := range friends {

		// for each direct friend find in other friend's list
		ofriends = friendships[directFriend]

		for _, key := range ofriends {

			// not itself
			if (key == username) {
				continue
			}

			// not direct friend
			if (hasElement(friends,key)) {
				continue
			}

			// add if not present in list
			if (! hasElement(mutualFriends,key)) {
				mutualFriends = append(mutualFriends, key)
			}
		}
	}

	return 	mutualFriends
}

