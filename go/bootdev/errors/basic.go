package errors

import (
	"fmt"
)

func sendSMSToCouple(msgToCustomer, msgToSpouse string) (int, error) {
	// ?
	cost , error := sendSMS(msgToCustomer);
	if (cost == 0) { 
		return 0, error
	}

	cost2 , error := sendSMS(msgToSpouse);
	if (cost2 == 0) {
		return 0, error
	}

	return cost + cost2, fmt.Errorf("")
}

// don't edit below this line

func sendSMS(message string) (int, error) {
	const maxTextLen = 25
	const costPerChar = 2
	if len(message) > maxTextLen {
		return 0, fmt.Errorf("can't send texts over %v characters", maxTextLen)
	}
	return costPerChar * len(message), nil
}
