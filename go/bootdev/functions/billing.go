package functions

func calculateFinalBill(costPerMessage float64, numMessages int) float64 {
	baseBill := calculateBaseBill(costPerMessage, numMessages)
	discountPercentage := calculateDiscount(numMessages)
	discountAmount := baseBill * discountPercentage
	finalBill := baseBill - discountAmount
	return finalBill
}

func calculateDiscount(messagesSent int) float64 {
	switch {
	case messagesSent > 5000:
		return 0.20
	case messagesSent > 1000:
		return 0.10
	default:
		return 0.0
	}
}

// don't touch below this line

func calculateBaseBill(costPerMessage float64, messagesSent int) float64 {
	return costPerMessage * float64(messagesSent)
}
