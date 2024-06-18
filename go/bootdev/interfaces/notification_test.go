package interfaces

import (
	"fmt"
	"strconv"
	"testing"
)

func Test(t *testing.T) {
	tests := []struct {
		notification       notification
		expectedID         string
		expectedImportance int
	}{
		{
			directMessage{senderUsername: "Kaladin", messageContent: "Life before death", priorityLevel: 10, isUrgent: true},
			"Kaladin",
			50,
		},
		{
			groupMessage{groupName: "Bridge 4", messageContent: "Soups ready!", priorityLevel: 2},
			"Bridge 4",
			2,
		},
		{
			systemAlert{alertCode: "ALERT001", messageContent: "THIS IS NOT A TEST HIGH STORM COMING SOON"},
			"ALERT001",
			100,
		},
	}
	if true {
		tests = append(tests,
			struct {
				notification       notification
				expectedID         string
				expectedImportance int
			}{
				directMessage{senderUsername: "Shallan", messageContent: "I am that I am.", priorityLevel: 5, isUrgent: false},
				"Shallan",
				5,
			},
			struct {
				notification       notification
				expectedID         string
				expectedImportance int
			}{
				groupMessage{groupName: "Knights Radiant", messageContent: "For the greater good.", priorityLevel: 10},
				"Knights Radiant",
				10,
			},
			struct {
				notification       notification
				expectedID         string
				expectedImportance int
			}{
				directMessage{senderUsername: "Adolin", messageContent: "Duels are my favorite.", priorityLevel: 3, isUrgent: true},
				"Adolin",
				50,
			},
		)
	}

	for i, test := range tests {
		t.Run("TestProcessNotification_"+strconv.Itoa(i+1), func(t *testing.T) {
			id, importance := processNotification(test.notification)
			if id != test.expectedID || importance != test.expectedImportance {
				t.Errorf("Test Failed:\nNotification: %+v\nExpected: %v/%d\nGot: %v/%d\n", test.notification, test.expectedID, test.expectedImportance, id, importance)
			} else {
				fmt.Printf("Test Passed:\nNotification: %+v\nExpected: %v/%d\nGot: %v/%d\n", test.notification, test.expectedID, test.expectedImportance, id, importance)
			}
		})
	}
}

