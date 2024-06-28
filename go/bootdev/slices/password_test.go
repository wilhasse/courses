package password

import (
	"testing"
)

func TestIsValidPassword(t *testing.T) {
	type testCase struct {
		password string
		isValid  bool
	}
	testCases := []testCase{
		{"Pass123", true},
		{"pas", false},
		{"Password", false},
		{"123456", false},
	}
	if true {
		testCases = append(testCases,
			[]testCase{
				{"VeryLongPassword1", false},
				{"Short", false},
				{"1234short", false},
				{"Short5", true},
				{"P4ssword", true},
			}...,
		)
	}

	for _, tc := range testCases {
		result := isValidPassword(tc.password)
		if result != tc.isValid {
			t.Errorf("Expected %v for password \"%s\", got %v", tc.isValid, tc.password, result)
		}
	}
}
