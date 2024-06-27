package rangeex

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		msg      []string
		badWords []string
		expected int
	}
	tests := []testCase{
		{[]string{"hey", "there", "john"}, []string{"crap", "shoot", "frick", "dang"}, -1},
		{[]string{"ugh", "oh", "my", "frick"}, []string{"crap", "shoot", "frick", "dang"}, 3},
	}
	if true {
		tests = append(tests, []testCase{
			{[]string{"what", "the", "shoot", "I", "hate", "that", "crap"}, []string{"crap", "shoot", "frick", "dang"}, 2},
			{[]string{"crap", "shoot", "frick", "dang"}, []string{""}, -1},
			{[]string{""}, nil, -1},
		}...)
	}

	for _, test := range tests {
		if output := indexOfFirstBadWord(test.msg, test.badWords); output != test.expected {
			t.Errorf(`Test Failed:
message:
%v
bad words:
%v
=>
expected:
  %v
actual:
  %v
===========================
`, sliceWithBullets(test.msg), sliceWithBullets(test.badWords), test.expected, output)
		} else {
			fmt.Printf(`Test Passed:
message:
%v
bad words:
%v
=>
expected:
  %v
actual:
  %v
===========================
`, sliceWithBullets(test.msg), sliceWithBullets(test.badWords), test.expected, output)
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

