package channels

import (
	"fmt"
	"slices"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		n        int
		expected []int
	}
	tests := []testCase{
		{5, []int{0, 1, 1, 2, 3}},
		{3, []int{0, 1, 1}},
	}
	if true {
		tests = append(tests, []testCase{
			{0, []int{}},
			{1, []int{0}},
			{7, []int{0, 1, 1, 2, 3, 5, 8}},
		}...)
	}

	for _, test := range tests {
		actual := concurrentFib(test.n)
		if !slices.Equal(actual, test.expected) {
			t.Errorf("Test Failed: (%v) ->\nexpected: %v\nactual: %v", test.n, test.expected, actual)
		} else {
			fmt.Printf("Test Passed: (%v) ->\nexpected: %v\nactual: %v\n", test.n, test.expected, actual)
		}
	}
}

