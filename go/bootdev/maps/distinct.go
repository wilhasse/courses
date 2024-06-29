package distinct

import (
	"strings"
)

func countDistinctWords(messages []string) int {
	// ?

	distinct := make(map[string]int)
	var words []string

	for _, msg := range messages {

		// split
		words = strings.Fields(msg)

		for _, word := range words {

			// first name caracter
		    _, ok := distinct[strings.ToLower(word)]
		    if !ok {
		        distinct[strings.ToLower(word)] = 0
		    }
		}
	}

	return len(distinct)
}
