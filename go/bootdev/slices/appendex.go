package appendex

//import "fmt"

/*
Example input / output

[]cost{
    {0, 4.0},
    {1, 2.1},
    {5, 2.5},
    {1, 3.1},
}
[]float64{
    4.0, // first day
    5.2, // 2.1 + 3.1
    0.0, // intermediate days with no costs
    0.0, // ...
    0.0, // ...
    2.5, // last day
} */

type cost struct {
	day   int
	value float64
}

func getCostsByDay(costs []cost) []float64 {
	// ?
	dados := make([]float64, 0)

	for i:=0;i<len(costs);i++ {

		//fmt.Printf("%d - %f",costs[i].day,costs[i].value)
		if (costs[i].day >= len(dados)) {

			// fill remaining with zeros
			remaning := make([]float64, costs[i].day-len(dados))
			dados = append(dados,remaning...)

			// new element
			dados = append(dados,costs[i].value)
		} else {
			dados[costs[i].day] += costs[i].value
		}
	}

	return dados
}
