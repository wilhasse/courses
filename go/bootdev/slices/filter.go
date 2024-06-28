package filter

//Your task is to implement a function that filters a slice of messages based on the message type.
//It should take a slice of Message interfaces and a string indicating the desired type ("text", "media", or "link"). 
//It should return a new slice of Message interfaces containing only messages of the specified type

type Message interface {
	Type() string
}

type TextMessage struct {
	Sender  string
	Content string
}

func (tm TextMessage) Type() string {
	return "text"
}

type MediaMessage struct {
	Sender    string
	MediaType string
	Content   string
}

func (mm MediaMessage) Type() string {
	return "media"
}

type LinkMessage struct {
	Sender  string
	URL     string
	Content string
}

func (lm LinkMessage) Type() string {
	return "link"
}

// Filter msgs with specific type
func filterMessages(messages []Message, filterType string) []Message {
	// ?
	ret := make([]Message, 0)
	for _, msg := range messages {
	    if (msg.Type() ==filterType) {
			ret = append(ret,msg)
		}
	}

	return ret
}
