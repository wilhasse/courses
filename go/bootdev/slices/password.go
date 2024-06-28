package password

import (
	"unicode"
)
func isValidPassword(password string) bool {
	// ?
	if (len(password) < 5) || (len(password) > 12) {
		return false
	}

	hasDigit := false
	hasUppercase := false
	for _, char := range password {

		if unicode.IsDigit(char) {
			hasDigit = true
		}
		if unicode.IsUpper(char) {
			hasUppercase = true
		}
		if hasDigit && hasUppercase {
			return true
		}
	}

	return false;
}
