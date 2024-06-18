package interfaces

type PlainText struct {
	message string
}

type Bold struct {
	message string
}

type Code struct {
	message string
}

type Formatter interface {
	Format() string
}

func (p PlainText) Format() string {
	return p.message
}

func (p Bold) Format() string {
	return "**" + p.message + "**"
}

func (p Code) Format() string {
	return "`" + p.message + "`"
}

// Don't Touch below this line

func SendMessage(formatter Formatter) string {
	return formatter.Format() // Adjusted to call Format without an argument
}
