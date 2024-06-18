package interfaces

import (
	"fmt"
	"strconv"
	"testing"
)

func TestSendMessage(t *testing.T) {
	tests := []struct {
		formatter Formatter
		expected  string
	}{
		{PlainText{message: "Hello, World!"}, "Hello, World!"},
		{Bold{message: "Bold Message"}, "**Bold Message**"},
		{Code{message: "Code Message"}, "`Code Message`"},
	}

	if true {
		tests = append(tests,
			struct {
				formatter Formatter
				expected  string
			}{Code{message: ""}, "``"},
			struct {
				formatter Formatter
				expected  string
			}{Bold{message: ""}, "****"},
			struct {
				formatter Formatter
				expected  string
			}{PlainText{message: ""}, ""},
		)
	}

	for i, test := range tests {
		testName := "Test Case " + strconv.Itoa(i+1)
		t.Run(testName, func(t *testing.T) {
			formattedMessage := SendMessage(test.formatter)
			if formattedMessage != test.expected {
				t.Errorf("%s\n Failed: Expected formatted message to be '%v', but got '%v'\n", testName, test.expected, formattedMessage)
			} else {
				fmt.Printf("%s\n Passed: Expected formatted message '%v' and got '%v'\n", testName, test.expected, formattedMessage)
			}
		})
	}
}
