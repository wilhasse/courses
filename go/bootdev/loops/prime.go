package main

import (
	"fmt"
)

func printPrimes(max int) {
	// ?
	for n:=2;n<=max;n++ {
		if n == 2 {
			fmt.Printf("%d\n", n)
			continue
		}

		if n % 2 == 0 {
			continue;
		}

		achou := false;
		for i:=3;i*i <= n;i++ {

			if n % i == 0 {
				achou = true;
			}
		}

		if (achou) {
			continue;
		}

		fmt.Printf("%d\n", n)
	}
}

// don't edit below this line

func test(max int) {
	fmt.Printf("Primes up to %v:\n", max)
	printPrimes(max)
	fmt.Println("===============================================================")
}

func main() {
	test(10)
	test(20)
	test(30)
}
