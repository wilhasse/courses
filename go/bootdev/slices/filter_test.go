package filter

import (
	"fmt"
	"testing"
)

func TestFilterMessages(t *testing.T) {
	messages := []Message{
		TextMessage{"Alice", "Hello, World!"},
		MediaMessage{"Bob", "image", "A beautiful sunset"},
		LinkMessage{"Charlie", "http://example.com", "Example Domain"},
		TextMessage{"Dave", "Another text message"},
		MediaMessage{"Eve", "video", "Cute cat video"},
		LinkMessage{"Frank", "https://boot.dev", "Learn Coding Online"},
	}

	testCases := []struct {
		filterType    string
		expectedCount int
		expectedType  string
	}{
		{"text", 2, "text"},
		{"media", 2, "media"},
		{"link", 2, "link"},
	}

	if true {
		testCases = append(testCases,
			struct {
				filterType    string
				expectedCount int
				expectedType  string
			}{"media", 2, "media"},
			struct {
				filterType    string
				expectedCount int
				expectedType  string
			}{"text", 2, "text"},
		)
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("TestCase%d", i+1), func(t *testing.T) {
			filtered := filterMessages(messages, tc.filterType)
			if len(filtered) != tc.expectedCount {
				t.Errorf("Test Case %d - Filtering for %s: expected %d messages, got %d", i+1, tc.filterType, tc.expectedCount, len(filtered))
			}

			for _, m := range filtered {
				if m.Type() != tc.expectedType {
					t.Errorf("Test Case %d - Expected a %s message, got %s", i+1, tc.expectedType, m.Type())
				}
			}
		})
	}
}

