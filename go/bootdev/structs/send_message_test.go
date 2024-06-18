package structs

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	tests := []struct {
		name           string
		membershipType membershipType
		message        string
		expectResult   string
		expectSuccess  bool
	}{
		{"Syl", TypeStandard, "Hello, Kaladin!", "Hello, Kaladin!", true},
		{"Pattern", TypePremium, "You are not as good with patterns... You are abstract. You think in lies and tell them to yourselves. That is fascinating, but it is not good for patterns.", "You are not as good with patterns... You are abstract. You think in lies and tell them to yourselves. That is fascinating, but it is not good for patterns.", true},
		{"Dalinar", TypeStandard, "I will take responsibility for what I have done. If I must fall, I will rise each time a better man.", "I will take responsibility for what I have done. If I must fall, I will rise each time a better man.", true},
	}
	if true {
		submitCases := []struct {
			name           string
			membershipType membershipType
			message        string
			expectResult   string
			expectSuccess  bool
		}{
			{"Pattern", TypeStandard, "Humans can see the world as it is not. It is why your lies can be so strong. You are able to not admit that they are lies.", "", false},
			{"Dabbid", TypePremium, ".........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................", "", false},
		}
		tests = append(tests, submitCases...)
	}

	for _, tc := range tests {
		user := newUser(tc.name, tc.membershipType)
		result, pass := user.SendMessage(tc.message, len(tc.message))
		if tc.expectSuccess != pass || result != tc.expectResult {
			t.Errorf("Test Failed for user %s with membership type %v\n  Sending message: '%s'\n  Expected success: %v, got: %v\n\n",
				tc.name, tc.membershipType, tc.message, tc.expectSuccess, pass)
		} else {
			fmt.Printf("Test Passed for user %s with membership type %v\n  Sending message: '%s'\n  Expected success: %v, Actual success: %v\n\n",
				tc.name, tc.membershipType, tc.message, tc.expectSuccess, pass)
		}
	}
}

