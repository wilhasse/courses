package main

/*
#include "add.h"
*/
import "C"
import (
    "fmt"
)

func main() {
    a, b := C.int(3), C.int(4)
    result := C.add(a, b)
    fmt.Printf("%d + %d = %d\n", a, b, int32(result))
}
