package channels

func concurrentFib(n int) []int {
	// ?
	dados := make([]int, 0)
	ch := make(chan int)
	go fibonacci(n,ch)
	for item := range ch {
		dados = append(dados,item)
	}
	return dados
}

// don't touch below this line

func fibonacci(n int, ch chan int) {
	x, y := 0, 1
	for i := 0; i < n; i++ {
		ch <- x
		x, y = y, x+y
	}
	close(ch)
}
