package matrix

func createMatrix(rows, cols int) [][]int {

	// Initialize matrix as a slice of slice of ints
	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {

		// Create a slice for each row
		row := make([]int, cols)
		for j := 0; j < cols; j++ {

			// Assign value to each element
			row[j] = i * j
		}

		// Assign the row to the matrix
		matrix[i] = row
	}
	return matrix
}
