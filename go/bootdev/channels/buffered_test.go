package channels

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		emails   []string
		expected int
	}
	tests := []testCase{
		{
			[]string{
				"To boldly go where no man has gone before.",
				"Live long and prosper.",
			},
			2,
		},
		{
			[]string{
				"The needs of the many outweigh the needs of the few, or the one.",
				"Change is the essential process of all existence.",
				"Resistance is futile.",
			},
			3,
		},
	}
	if true {
		tests = append(tests, []testCase{
			{
				[]string{
					"It's life, Jim, but not as we know it.",
					"Infinite diversity in infinite combinations.",
					"Make it so.",
					"Engage!",
				},
				4,
			},
		}...)
	}

	for _, test := range tests {
		ch := addEmailsToQueue(test.emails)
		actual := len(ch)
		if actual != test.expected {
			t.Errorf(`Test Failed:
%v
->
expected channel length: %v
actual channel length: %v
=======================
`,
				sliceWithBullets(test.emails),
				test.expected,
				actual,
			)
		} else {
			fmt.Printf(`Test Passed:
%v
->
expected channel length: %v
actual channel length: %v
=======================
`,
				sliceWithBullets(test.emails),
				test.expected,
				actual,
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

