package slices

import (
	"fmt"
	"slices"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		plan             string
		messages         [3]string
		expectedMessages []string
		expectedErr      string
	}
	tests := []testCase{
		{
			planFree,
			[3]string{
				"Hello sir/madam can I interest you in a yacht?",
				"Please I'll even give you an Amazon gift card?",
				"You're missing out big time",
			},
			[]string{"Hello sir/madam can I interest you in a yacht?", "Please I'll even give you an Amazon gift card?"},
			"",
		},
		{
			planPro,
			[3]string{
				"Hello sir/madam can I interest you in a yacht?",
				"Please I'll even give you an Amazon gift card?",
				"You're missing out big time",
			},
			[]string{
				"Hello sir/madam can I interest you in a yacht?",
				"Please I'll even give you an Amazon gift card?",
				"You're missing out big time",
			},
			"",
		},
	}
	if true {
		tests = append(tests, []testCase{
			{
				planFree,
				[3]string{
					"You can get a good look at a T-bone by sticking your head up a bull's ass, but wouldn't you rather take the butcher's word for it?",
					"Wouldn't you?",
					"Wouldn't you???",
				},
				[]string{
					"You can get a good look at a T-bone by sticking your head up a bull's ass, but wouldn't you rather take the butcher's word for it?",
					"Wouldn't you?",
				},
				"",
			},
			{
				planPro,
				[3]string{
					"You can get a good look at a T-bone by sticking your head up a bull's ass, but wouldn't you rather take the butcher's word for it?",
					"Wouldn't you?",
					"Wouldn't you???",
				},
				[]string{
					"You can get a good look at a T-bone by sticking your head up a bull's ass, but wouldn't you rather take the butcher's word for it?",
					"Wouldn't you?",
					"Wouldn't you???",
				},
				"",
			},
			{
				"invalid plan",
				[3]string{
					"You can get a good look at a T-bone by sticking your head up a bull's ass, but wouldn't you rather take the butcher's word for it?",
					"Wouldn't you?",
					"Wouldn't you???",
				},
				nil,
				"unsupported plan",
			},
		}...)
	}

	for _, test := range tests {
		actualMessages, err := getMessageWithRetriesForPlan(test.plan, test.messages)
		errString := ""
		if err != nil {
			errString = err.Error()
		}
		if !slices.Equal(actualMessages, test.expectedMessages) || errString != test.expectedErr {
			t.Errorf(`Test Failed:
Plan: %v
Messages:
%v
=>
expected:
%v
errString: %v
actual:
%v
errString: %v
===========================
	`,
				test.plan,
				sliceWithBullets(test.messages[:]),
				sliceWithBullets(test.expectedMessages),
				test.expectedErr,
				sliceWithBullets(actualMessages),
				errString,
			)
		} else {
			fmt.Printf(`Test Passed:
Plan: %v
Messages:
%v
=>
expected:
%v
errString: %v
actual:
%v
errString: %v
===========================
	`,
				test.plan,
				sliceWithBullets(test.messages[:]),
				sliceWithBullets(test.expectedMessages),
				test.expectedErr,
				sliceWithBullets(actualMessages),
				errString,
			)
		}
	}
}

func sliceWithBullets[T any](slice []T) string {
	if slice == nil {
		return "  <nil>"
	}
	if len(slice) == 0 {
		return "  []"
	}
	output := ""
	for i, item := range slice {
		form := "  - %#v\n"
		if i == (len(slice) - 1) {
			form = "  - %#v"
		}
		output += fmt.Sprintf(form, item)
	}
	return output
}
