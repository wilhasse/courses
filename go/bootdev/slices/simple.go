package slices

import (
	"errors"
	)

const (
	planFree = "free"
	planPro  = "pro"
)

func getMessageWithRetriesForPlan(plan string, messages [3]string) ([]string, error) {
	// ?
	if (plan == "pro") {
		return messages[:],nil
	}

	if (plan == "free") {
		return messages[0:2],nil
	}

	var err error = errors.New("unsupported plan")
	return nil,err
}
