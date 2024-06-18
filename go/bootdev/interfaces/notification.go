package interfaces

type notification interface {
	importance() int
}

type directMessage struct {
	senderUsername string
	messageContent string
	priorityLevel  int
	isUrgent       bool
}

type groupMessage struct {
	groupName      string
	messageContent string
	priorityLevel  int
}

type systemAlert struct {
	alertCode      string
	messageContent string
}

// ?
func (d directMessage) importance() int {
	if d.isUrgent {
		return 50
	} else {
		return d.priorityLevel
	}
}

func (d groupMessage) importance() int {
	return d.priorityLevel
}

func (d systemAlert) importance() int {
	return 100
}

func processNotification(n notification) (string, int) {
	// ?
	c, ok := n.(directMessage)
	if (ok) {
		return c.senderUsername,c.importance()
	}

	g, ok := n.(groupMessage)
	if (ok) {
		return g.groupName,g.importance()
	}

	s, ok := n.(systemAlert)
	if (ok) {
		return s.alertCode,s.importance()
	}

	return "",0

}
