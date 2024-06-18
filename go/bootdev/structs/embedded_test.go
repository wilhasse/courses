package structs

import (
	"fmt"
	"testing"
)

func getSenderLog(s sender) string {
	return fmt.Sprintf(`
====================================
Sender name: %v
Sender number: %v
Sender rateLimit: %v
====================================
`, s.name, s.number, s.rateLimit)
}

func Test(t *testing.T) {
	type testCase struct {
		rateLimit int
		name      string
		number    int
		expected  string
	}
	tests := []testCase{
		{
			10000,
			"Deborah",
			18055558790,
			`
====================================
Sender name: Deborah
Sender number: 18055558790
Sender rateLimit: 10000
====================================
`,
		},
		{
			5000,
			"Jason",
			18055558791,
			`
====================================
Sender name: Jason
Sender number: 18055558791
Sender rateLimit: 5000
====================================
`,
		},
	}
	if true {
		tests = append(tests, []testCase{
			{
				1000,
				"Jill",
				18055558792,
				`
====================================
Sender name: Jill
Sender number: 18055558792
Sender rateLimit: 1000
====================================
`,
			},
		}...)
	}

	for _, test := range tests {
		output := getSenderLog(sender{
			rateLimit: test.rateLimit,
			user: user{
				name:   test.name,
				number: test.number,
			},
		})
		if output != test.expected {
			t.Errorf(
				"Test Failed: (%v, %v, %v) -> expected: %v actual: %v",
				test.rateLimit,
				test.name,
				test.number,
				test.expected,
				output,
			)
		} else {
			fmt.Printf("Test Passed: (%v, %v, %v) -> expected: %v actual: %v\n",
				test.rateLimit,
				test.name,
				test.number,
				test.expected,
				output,
			)
		}
	}
}

