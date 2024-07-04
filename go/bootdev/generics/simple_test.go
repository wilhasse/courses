package generics

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	test(
		t,
		[]int{1, 2, 3, 4},
		4,
	)
	test(
		t,
		[]string{"a", "b", "c", "d"},
		"d",
	)
	if true {
		test(
			t,
			[]int{},
			0,
		)
		test(
			t,
			[]bool{true, false, true, true, false},
			false,
		)
	}

}

func test[T comparable](t *testing.T, s []T, expected T) {
	if output := getLast(s); output != expected {
		t.Errorf(
			"Test Failed: (%v) -> expected: %v actual: %v",
			s,
			expected,
			output,
		)
	} else {
		fmt.Printf("Test Passed: (%v) -> expected: %v actual: %v\n",
			s,
			expected,
			output,
		)
	}
}
