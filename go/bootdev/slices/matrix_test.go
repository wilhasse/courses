package matrix

import (
	"fmt"
	"reflect"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		rows, cols int
		expected   [][]int
	}
	tests := []testCase{
		{3, 3, [][]int{
			{0, 0, 0},
			{0, 1, 2},
			{0, 2, 4},
		}},
		{4, 4, [][]int{
			{0, 0, 0, 0},
			{0, 1, 2, 3},
			{0, 2, 4, 6},
			{0, 3, 6, 9},
		}},
	}
	if true {
		tests = append(tests, []testCase{
			{5, 7, [][]int{
				{0, 0, 0, 0, 0, 0, 0},
				{0, 1, 2, 3, 4, 5, 6},
				{0, 2, 4, 6, 8, 10, 12},
				{0, 3, 6, 9, 12, 15, 18},
				{0, 4, 8, 12, 16, 20, 24},
			}},
			{0, 0, [][]int{}},
		}...)
	}

	for _, test := range tests {
		if output := createMatrix(test.rows, test.cols); !reflect.DeepEqual(output, test.expected) {
			t.Errorf(`Test Failed: %v x %v matrix ->
	expected:
%v
	actual:
%v
`,
				test.rows, test.cols, formatMatrix(test.expected), formatMatrix(output))
		} else {
			fmt.Printf(`Test Passed: %v x %v matrix ->
	expected:
%v
	actual:
%v
`,
				test.rows, test.cols, formatMatrix(test.expected), formatMatrix(output))
		}
	}
}

func formatMatrix(matrix [][]int) string {
	var result string
	for _, row := range matrix {
		result += fmt.Sprintf("%v\n", row)
	}
	return result
}
