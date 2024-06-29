package count

func getCounts(messagedUsers []string, validUsers map[string]int) {
	// ?
	for _, user := range messagedUsers {

		/*
		It doesn't work, because vuser is a copy

		if vuser, ok := validUsers[user]; ok {
		    vuser = vuser + 1
		}
		*/

		// correct
		if _, ok := validUsers[user]; ok {
		    validUsers[user] += 1
		}
	}
}
