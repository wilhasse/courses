package channels

import (
	"fmt"
	"testing"
)

func Test(t *testing.T) {
	type testCase struct {
		numDBs int
	}
	tests := []testCase{
		{1},
		{3},
		{4},
	}
	if true {
		tests = append(tests, []testCase{
			{0},
			{13},
		}...)
	}

	for _, test := range tests {
		fmt.Printf("Testing %v Databases...\n", test.numDBs)
		dbChan, count := getDBsChannel(test.numDBs)
		waitForDBs(test.numDBs, dbChan)
		for *count != test.numDBs {
			fmt.Println("...")
		}
		fmt.Printf(`Expected - length: %v, count: %v
  Actual - length: %v, count: %v
Passed
=======================
`,
			0,
			test.numDBs,
			len(dbChan),
			*count,
		)
	}
}

