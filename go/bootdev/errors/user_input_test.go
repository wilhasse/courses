package errors

import (
	"fmt"
	"testing"
)

func TestValidateStatus(t *testing.T) {
	testCases := []struct {
		status      string
		expectedErr string
	}{
		{"", "status cannot be empty"},
		{"This is a valid status update that is well within the character limit.", ""},
		{"This status update is way too long. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco.", "status exceeds 140 characters"},
	}
	if true {
		testCases = append(testCases,
			struct {
				status      string
				expectedErr string
			}{"Another valid status.", ""},
			struct {
				status      string
				expectedErr string
			}{"This status update, while derivative, contains exactly one hundred and forty-one characters, which is over the status update character limit.", "status exceeds 140 characters"},
		)
	}

	for _, tc := range testCases {
		err := validateStatus(tc.status)
		errString := ""
		if err != nil {
			errString = err.Error()
		}
		if errString != tc.expectedErr {
			t.Errorf(`Test Failed: "%v" =>
    expected error: "%v"
    actual error: "%v"
`, tc.status, tc.expectedErr, errString)
		} else {
			fmt.Printf(`Test Passed: "%v" =>
    expected error: "%v"
    actual error: "%v"
`, tc.status, tc.expectedErr, errString)
		}
	}
}

