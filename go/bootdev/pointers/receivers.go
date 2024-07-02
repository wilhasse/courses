package main

import "fmt"

type car struct {
	color string
}

func (c *car) setColor(color string) {
	c.color = color
}

type car2 struct {
	color string
}

func (c car2) setColor(color string) {
	c.color = color
}

func main() {
	c := car{
		color: "white",
	}
	c.setColor("blue")
	fmt.Println(c.color)
	// prints "blue"

        c2 := car2{
                color: "white",
        }
        c2.setColor("blue")
        fmt.Println(c2.color)
        // prints "white"
}
