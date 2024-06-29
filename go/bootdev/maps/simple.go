package simple

import (
	"errors"
)

func getUserMap(names []string, phoneNumbers []int) (map[string]user, error) {

	// ?
	if (len(names) != len(phoneNumbers)) {
		return nil,errors.New("invalid sizes")
	}

	ret := make(map[string]user)

	for i := 0; i < len(names); i++ {

		ret[names[i]] = user {name: names[i], phoneNumber: phoneNumbers[i]}
	}

	return ret,nil
}

type user struct {
	name        string
	phoneNumber int
}
