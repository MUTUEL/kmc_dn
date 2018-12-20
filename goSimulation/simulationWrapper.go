package main

import "C"
import "fmt"
import "sync"
//import ".simulation"

var count int
var mtx sync.Mutex

func Log(msg string) int {
	mtx.Lock()
	defer mtx.Unlock()
	fmt.Println(msg)
	count++
	return count
  }
  
func deFlattenFloat64(m []float64, x int64, y int64) [][]float64 {
	  r := make([][]float64, x)
	  c := 0
	  for i := int64(0); i < x; i+=1 {
		  r[i] = make([]float64, y)
		  for j := int64(0); j < y; j+=1 {
			  r[i][j] = m[c]
			  c += 1
		  }
	  }
	  Log(fmt.Sprintf("%v", r))
	  return r
}

//export simulateWrapper
func simulateWrapper(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, time float64,
		occupation []bool, distances []float64, E_constant []float64, transitions_constant float64,
		electrode_occupation []int, site_energies []float64, hops int) float64{
	newDistances := deFlattenFloat64(distances, NSites, NSites);
	return simulate(int(NSites), int(NElectrodes), nu, kT, I_0, R, time, occupation, 
		newDistances , E_constant, transitions_constant, electrode_occupation, site_energies, hops)
}