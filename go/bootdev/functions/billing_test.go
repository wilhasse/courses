package functions

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		costPerMessage float64
		numMessages    int
		expected       float64
	}
	tests := []testCase{
		{2.55, 89, 226.95},
		{2.25, 204, 459},
		{1, 1428, 1285.2},
		{5, 1000, 5000},
		{5, 1001, 4504.5},
	}
	if true {
		tests = append(tests, []testCase{
			{3, 0, 0},
			{3, 7421, 17810.4},
		}...)
	}

	for _, test := range tests {
		if output := calculateFinalBill(
			test.costPerMessage,
			test.numMessages,
		); output != test.expected {
			t.Errorf(
				"Test Failed: (%v, %v) -> expected: %v actual: %v",
				test.costPerMessage,
				test.numMessages,
				test.expected,
				output,
			)
		} else {
			fmt.Printf("Test Passed: (%v, %v) -> expected: %v actual: %v\n",
				test.costPerMessage,
				test.numMessages,
				test.expected,
				output,
			)
		}
	}
}

