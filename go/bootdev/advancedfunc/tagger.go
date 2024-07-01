package advancedfunc

import (
	"fmt"
	"strings"
)

type sms struct {
	id      string
	content string
	tags    []string
}

func tagMessages(messages []sms, tagger func(sms) []string) []sms {
	// ?
	/*
	This tricky: it copy msg and it doesn't modify tags
	It doesn't work, the solution is to have an variable msg
	but iterate directly using messages

	wrong:
	for _, msg := range messages {
		msg.tags = tagger(msg)
	}
	*/
	// correct
	for i := range messages {
		messages[i].tags = tagger(messages[i])
	}
	return messages
}

func tagger(msg sms) []string {
	tags := []string{}
	// ?
	fmt.Println(msg.content)
	if strings.Contains(msg.content,"Urgent") || strings.Contains(msg.content,"urgent") {

		tags = append(tags,"Urgent")
	}
	if strings.Contains(msg.content,"Sale") || strings.Contains(msg.content,"sale") {

		tags = append(tags,"Promo")
	}

	return tags
}
