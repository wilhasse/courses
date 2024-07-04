package generics

func getLast[T any](s []T) T {
	// ?
	var myZero T

	if (len(s) == 0) {
		return myZero
	}

	return s[len(s)-1]
}
