package friends

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func Test(t *testing.T) {
	friendships := map[string][]string{
		"Dalinar": {"Kaladin", "Pattern"},
		"Kaladin":  {"Dalinar", "Syl", "Teft"},
		"Pattern":  {"Dalinar", "Teft"},
		"Syl":      {"Kaladin"},
		"Teft":     {"Kaladin", "Pattern"},
		"Moash":    {},
		"Shallan":  {"Pattern", "Kaladin"},
	}

	testCases := []struct {
		username string
		expected []string
	}{
		{"Dalinar", []string{"Syl", "Teft"}},
		{"Kaladin", []string{"Pattern"}},
		{"Pattern", []string{"Kaladin"}},
		{"Syl", []string{"Dalinar", "Teft"}},
		{"Teft", []string{"Dalinar", "Syl"}},
		{"Moash", nil},
	}

	if true {
		testCases = append(testCases, struct {
			username string
			expected []string
		}{
			"Odium", nil,
		},
			struct {
				username string
				expected []string
			}{
				"Shallan", []string{"Dalinar", "Syl", "Teft"},
			},
		)
	}

	for _, tc := range testCases {
		t.Run(tc.username, func(t *testing.T) {
			result := findSuggestedFriends(tc.username, friendships)
			sort.Strings(result)
			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("\nTest Failed for username %s:\n  Expected mutual friends to be %s,\n  Got %s\n",
					tc.username, formatSlice(tc.expected), formatSlice(result))
			} else {
				fmt.Printf("\nTest Passed for username %s:\n  Expected mutual friends: %s\n  Actual mutual friends: %s\n",
					tc.username, formatSlice(tc.expected), formatSlice(result))
			}
		})
	}
}

func formatSlice(slice []string) string {
	if slice == nil {
		return "nil"
	}
	return fmt.Sprintf("%+v", slice)
}

