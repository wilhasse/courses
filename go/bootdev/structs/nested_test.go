package structs

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		mToSend  messageToSend
		expected bool
	}
	tests := []testCase{
		{messageToSend{
			message:   "you have an appointment tomorrow",
			sender:    user{name: "Brenda Halafax", number: 16545550987},
			recipient: user{name: "Sally Sue", number: 19035558973},
		}, true},
		{messageToSend{
			message:   "you have an event tomorrow",
			sender:    user{number: 16545550987},
			recipient: user{name: "Suzie Sall", number: 19035558973},
		}, false},
	}
	if true {
		tests = append(tests, []testCase{
			{messageToSend{
				message:   "you have an birthday tomorrow",
				sender:    user{name: "Jason Bjorn", number: 16545550987},
				recipient: user{name: "Jim Bond"},
			}, false},
			{messageToSend{
				message:   "you have a party tomorrow",
				sender:    user{name: "Njorn Halafax"},
				recipient: user{name: "Becky Sue", number: 19035558973},
			}, false},
			{messageToSend{
				message:   "you have a birthday tomorrow",
				sender:    user{name: "Eli Halafax", number: 16545550987},
				recipient: user{number: 19035558973},
			}, false},
		}...)
	}

	for _, test := range tests {
		if output := canSendMessage(test.mToSend); output != test.expected {
			t.Errorf(
				"Test Failed: (%+v) -> expected: %v actual: %v",
				test.mToSend,
				test.expected,
				output,
			)
		} else {
			fmt.Printf("Test Passed: (%+v) -> expected: %v actual: %v\n", test.mToSend, test.expected, output)
		}
	}
}

