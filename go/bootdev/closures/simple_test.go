package closures

import (
	"fmt"
	"slices"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		input    []int
		expected []int
	}
	tests := []testCase{
		{
			input:    []int{1, 2, 3},
			expected: []int{1, 3, 6},
		},
		{
			input:    []int{1, 2, 3, 4, 5},
			expected: []int{1, 3, 6, 10, 15},
		},
	}

	if true {
		tests = append(tests, []testCase{
			{
				input:    []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
				expected: []int{1, 3, 6, 10, 15, 21, 28, 36, 45, 55},
			},
			{
				input:    []int{0, 0, 0, 0},
				expected: []int{0, 0, 0, 0},
			},
			{
				input:    []int{5, -3, -1},
				expected: []int{5, 2, 1},
			},
		}...)
	}

	for _, test := range tests {
		f := adder()
		results := make([]int, len(test.input))
		for i, v := range test.input {
			results[i] = f(v)
		}
		if !slices.Equal(results, test.expected) {
			t.Errorf(`
Test Failed.
  input: %v
->
  expected: %v
  actual: %v
`,
				test.input,
				test.expected,
				results,
			)
		} else {
			fmt.Printf(`
Test Passed.
  input: %v
->
  expected: %v
  actual: %v
			`,
				test.input,
				test.expected,
				results,
			)
		}
	}
}

