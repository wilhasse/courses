package advancedfunc

import (
	"fmt"
	"reflect"
	"testing"
)

func TestTagMessages(t *testing.T) {
	tests := []struct {
		messages []sms
		expected [][]string
	}{
		{
			messages: []sms{{id: "001", content: "Urgent, please respond!"}, {id: "002", content: "Big sale on all items!"}},
			expected: [][]string{{"Urgent"}, {"Promo"}},
		},
		{
			messages: []sms{{id: "003", content: "Enjoy your day"}},
			expected: [][]string{{}},
		},
	}

	if true {
		tests = append(tests, struct {
			messages []sms
			expected [][]string
		}{
			messages: []sms{{id: "004", content: "Sale! Don't miss out on these urgent promotions!"}},
			expected: [][]string{{"Urgent", "Promo"}},
		})
	}

	for _, test := range tests {
		actual := tagMessages(test.messages, tagger)
		for i, msg := range actual {
			if !reflect.DeepEqual(msg.tags, test.expected[i]) {
				t.Errorf("Test Failed for message ID %s.\n Expected tags: %v\n Actual tags: %v\n", msg.id, test.expected[i], msg.tags)
			} else {
				fmt.Printf("Test Passed for message ID %s.\n Expected tags: %v\n Actual tags: %v\n", msg.id, test.expected[i], msg.tags)
			}
		}
	}
}
