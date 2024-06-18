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
			email{isSubscribed: true, body: "Whoa there!", toAddress: "soldier@monty.com"},
			"soldier@monty.com",
			0.11,
		},
		{
			sms{isSubscribed: false, body: "Halt! Who goes there?", toPhoneNumber: "+155555509832"},
			"+155555509832",
			2.1,
		},
	}
	if true {
		tests = append(tests, []testCase{
			{
				email{
					isSubscribed: false,
					body:         "It is I, Arthur, son of Uther Pendragon, from the castle of Camelot. King of the Britons, defeator of the Saxons, sovereign of all England!",
					toAddress:    "soldier@monty.com",
				},
				"soldier@monty.com",
				6.95,
			},
			{
				email{
					isSubscribed: true,
					body:         "Pull the other one!",
					toAddress:    "arthur@monty.com",
				},
				"arthur@monty.com",
				0.19,
			},
			{
				sms{
					isSubscribed:  true,
					body:          "I am. And this my trusty servant Patsy.",
					toPhoneNumber: "+155555509832",
				},
				"+155555509832",
				1.17,
			},
			{
				invalid{},
				"",
				0.0,
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

