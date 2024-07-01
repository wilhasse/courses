package pointers

import "fmt"

func main() {
	var x int = 50
	var y *int = &x
	*y = 100
	fmt.Printf("address of y %d\n",y)
	fmt.Printf("value of y %d\n",*y)
	fmt.Printf("value of x %d\n",x)
}
