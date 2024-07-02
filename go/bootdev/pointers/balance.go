package pointers

import (
	"errors"
)

type customer struct {
	id      int
	balance float64
}

type transactionType string

const (
	transactionDeposit    transactionType = "deposit"
	transactionWithdrawal transactionType = "withdrawal"
)

type transaction struct {
	customerID      int
	amount          float64
	transactionType transactionType
}

// Don't touch above this line

// ?
func updateBalance(customer *customer, transaction transaction) error {

	if transaction.transactionType == "deposit" {

		customer.balance += transaction.amount
		return nil
	}

	if transaction.transactionType == "withdrawal" {

		if (transaction.amount > customer.balance) {
			return errors.New("insufficient funds")
		}
		customer.balance -= transaction.amount
		return nil
	}

	return errors.New("unknown transaction type")
}
