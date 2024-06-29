package mutations

import (
	"testing"
)

func TestDeleteIfNecessary(t *testing.T) {
	type testCase struct {
		users             map[string]user
		name              string
		expectedErrString string
		expectedUsers     map[string]user
		expectedDeleted   bool
	}
	tests := []testCase{
		{
			map[string]user{"Erwin": {"Erwin", 14355550987, true}, "Levi": {"Levi", 98765550987, true}, "Hanji": {"Hanji", 18265554567, true}},
			"Erwin",
			"",
			map[string]user{"Levi": {"Levi", 98765550987, true}, "Hanji": {"Hanji", 18265554567, true}},
			true,
		},
		{
			map[string]user{"Erwin": {"Erwin", 14355550987, false}, "Levi": {"Levi", 98765550987, false}, "Hanji": {"Hanji", 18265554567, false}},
			"Erwin",
			"",
			map[string]user{"Erwin": {"Erwin", 14355550987, false}, "Levi": {"Levi", 98765550987, false}, "Hanji": {"Hanji", 18265554567, false}},
			false,
		},
	}
	if true {
		tests = append(tests, []testCase{
			{
				map[string]user{"Erwin": {"Erwin", 14355550987, true}, "Levi": {"Levi", 98765550987, true}, "Hanji": {"Hanji", 18265554567, true}},
				"Eren",
				"not found",
				map[string]user{"Erwin": {"Erwin", 14355550987, true}, "Levi": {"Levi", 98765550987, true}, "Hanji": {"Hanji", 18265554567, true}},
				false,
			},
		}...)
	}

	for _, test := range tests {
		deleted, err := deleteIfNecessary(test.users, test.name)
		if test.expectedErrString != "" {
			if err == nil {
				t.Errorf("Test Failed: expected error but got none")
			} else if err.Error() != test.expectedErrString {
				t.Errorf("Test Failed: expected error %v but got %v", test.expectedErrString, err)
			}
		} else if err != nil {
			t.Errorf("Test Failed: expected no error but got %v", err)
		} else if !compareMaps(test.users, test.expectedUsers) {
			t.Errorf(
				"Test Failed: \nexpected users: %v \nactual: %v",
				test.expectedUsers,
				test.users,
			)
		} else if deleted != test.expectedDeleted {
			t.Errorf(
				"Test Failed: \nexpected deleted: %v \nactual: %v",
				test.expectedDeleted,
				deleted,
			)
		}
	}
}

func compareMaps(map1, map2 map[string]user) bool {
	if len(map1) != len(map2) {
		return false
	}
	for key, value1 := range map1 {
		if value2, exist := map2[key]; !exist || value1 != value2 {
			return false
		}
	}
	return true
}

