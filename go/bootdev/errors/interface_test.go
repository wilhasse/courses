package errors

import (
	"fmt"
	"testing"
)

func TestDivide(t *testing.T) {
	type testCase struct {
		dividend, divisor, expected float64
		expectedError               string
	}
	tests := []testCase{
		{10, 2, 5, ""},
		{15, 3, 5, ""},
	}
	if true {
		tests = append(tests, []testCase{
			{10, 0, 0, "can not divide 10 by zero"},
			{15, 0, 0, "can not divide 15 by zero"},
			{100, 10, 10, ""},
			{16, 4, 4, ""},
			{30, 6, 5, ""},
		}...)
	}

	for _, test := range tests {
		output, err := divide(test.dividend, test.divisor)
		var errString string
		if err != nil {
			errString = err.Error()
		}
		if output != test.expected || errString != test.expectedError {
			t.Errorf(
				"Test Failed: (%v, %v) -> expected: (%v, %v), actual: (%v, %v)",
				test.dividend,
				test.divisor,
				test.expected,
				test.expectedError,
				output,
				errString,
			)
		} else {
			fmt.Printf("Test Passed: (%v, %v) -> expected: (%v, %v), actual: (%v, %v)\n",
				test.dividend,
				test.divisor,
				test.expected,
				test.expectedError,
				output,
				errString,
			)
		}
	}
}
