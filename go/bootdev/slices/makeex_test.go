package makeex

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		messages    []string
		expected    []float64
		expectedCap int
	}
	tests := []testCase{
		{
			[]string{"Welcome to the movies!", "Enjoy your popcorn!"},
			[]float64{0.22, 0.19},
			2,
		},
		{
			[]string{"I don't want to be here anymore", "Can we go home?", "I'm hungry", "I'm bored"},
			[]float64{0.31, 0.15, 0.1, 0.09},
			4,
		},
	}
	if true {
		tests = append(tests, []testCase{
			{[]string{}, []float64{}, 0},
			{[]string{""}, []float64{0}, 1},
			{[]string{"Hello", "Hi", "Hey"}, []float64{0.05, 0.02, 0.03}, 3},
		}...)
	}

	for _, test := range tests {
		if output := getMessageCosts(test.messages); !slicesEqual(output, test.expected) || cap(output) != test.expectedCap {
			t.Errorf(`Test Failed:
%v
=>
expected:
%v
expected cap: %v
actual:
%v
actual cap: %v
===========================
`,
				sliceWithBullets(test.messages),
				sliceWithBullets(test.expected),
				test.expectedCap,
				sliceWithBullets(output),
				cap(output),
			)
		} else {
			fmt.Printf(`Test Passed:
%v
=>
expected:
%v
expected cap: %v
actual:
%v
actual cap: %v
===========================
			`,
				sliceWithBullets(test.messages),
				sliceWithBullets(test.expected),
				test.expectedCap,
				sliceWithBullets(output),
				cap(output),
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

func slicesEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}
