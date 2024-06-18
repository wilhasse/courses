package interfaces

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		expense      expense
		expectedTo   string
		expectedCost float64
	}
	tests := []testCase{
		{
			email{true, "hello there", "kit@boga.com"},
			"kit@boga.com",
			0.11,
		},
		{
			sms{false, "I'm a Nigerian prince, please send me your bank info so I can deposit $1000 dollars", "+155555509832"},
			"+155555509832",
			8.3,
		},
	}

	if true {
		tests = append(tests, []testCase{
			{invalid{}, "", 0},
			{
				email{false, "This meeting could have been an email", "jane@doe.com"},
				"jane@doe.com",
				1.85,
			},
			{
				sms{false, "Please sir/madam", "+155555504444"},
				"+155555504444",
				1.6,
			},
		}...)
	}

	for _, test := range tests {
		if to, cost := getExpenseReport(test.expense); to != test.expectedTo || cost != test.expectedCost {
			t.Errorf(
				`Test Failed: %+v ->
	expected: (%v, %v)
	actual: (%v, %v)
`,
				test.expense,
				test.expectedTo,
				test.expectedCost,
				to,
				cost,
			)
		} else {
			fmt.Printf(
				`Test Passed: %+v ->
	expected: (%v, %v)
	actual: (%v, %v)
`,
				test.expense,
				test.expectedTo,
				test.expectedCost,
				to,
				cost,
			)
		}
	}
}

