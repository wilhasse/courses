package nested

import "unicode/utf8"

/*
For example:

billy
billy
bob
joe
Copy icon
Creates the following nested map:

b: {
    billy: 2,
    bob: 1
},
j: {
    joe: 1
}
*/

func getNameCounts(names []string) map[rune]map[string]int {

	counts := make(map[rune]map[string]int)

	for _, name := range names {

		// Get the first rune of the name
		firstRune, size := utf8.DecodeRuneInString(name)
		if size == 0 {
			continue // skip invalid runes
		}

		// first name caracter
	    letter, ok := counts[firstRune]
	    if !ok {
	        letter = make(map[string]int)
	        counts[firstRune] = letter
	    }

		// add count
		_, ok = letter[name]
	    if !ok {
			letter[name] = 1
		} else {
		    letter[name]++
		}
	}

	return counts
}
