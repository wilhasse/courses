package makeex

func getMessageCosts(messages []string) []float64 {
	// ?
	my := make([]float64, len(messages))

	for i:=0; i<len(messages); i++ {
		my[i] = float64(len(messages[i])) * 0.01
	}

	return my
}
