package channels

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		numBatches int
		expected   int
	}
	tests := []testCase{
		{3, 114},
		{4, 198},
	}
	if true {
		tests = append(tests, []testCase{
			{0, 0},
			{1, 15},
			{6, 435},
		}...)
	}

	for _, test := range tests {
		numSentCh := make(chan int)
		go sendReports(test.numBatches, numSentCh)
		output := countReports(numSentCh)
		if output != test.expected {
			t.Errorf(
				`Test Failed:
numBatches: %v
->
expected: %v
actual: %v
`,
				test.numBatches,
				test.expected,
				output,
			)
		} else {
			fmt.Printf(
				`Test Passed:
numBatches: %v
->
expected: %v
actual: %v
`,
				test.numBatches,
				test.expected,
				output,
			)
		}
	}
}

