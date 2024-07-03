package channels

func countReports(numSentCh chan int) int {
	// ?
	size := 0;
	for {
		total ,ok := <- numSentCh 
		if ! ok {
			return size
		}
		size += total
	}

	return size
}

// don't touch below this line

func sendReports(numBatches int, ch chan int) {
	for i := 0; i < numBatches; i++ {
		numReports := i*23 + 32%17
		ch <- numReports
	}
	close(ch)
}
