package main

import "C"
import "fmt"
import "sync"
import "math"
import "math/rand"
import (
	"io/ioutil" 
    "encoding/json"
)

var count int
var mtx sync.Mutex

var channelsMap map[int](chan float64) = make(map[int](chan float64))
var channelMapIndex int = 0

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
	return r
}

func deFlattenFloatTo32(m []float64, x int64, y int64) [][]float32 {
	r := make([][]float32, x)
	c := 0
	for i := int64(0); i < x; i+=1 {
		r[i] = make([]float32, y)
		for j := int64(0); j < y; j+=1 {
			r[i][j] = float32(m[c])
			c += 1
		}
	}
	return r
}

func toFloat32(m []float64) []float32{
	r := make([]float32, len(m))
	for i := 0; i < len(m); i++ {
		r[i] = float32(m[i])
	}
	return r
}

func logListToJson(list_f []float64, file_name string) {
    list_json, _ := json.Marshal(list_f)
    _ = ioutil.WriteFile(file_name, list_json, 0644)
}

func printAverageExpRandom(){
	list_f := make([]float64, 100000)
	sum := 0.0
	for i := 0; i < 100000; i++ {
		r_f := rand.ExpFloat64()/0.23
		list_f[i] = r_f
		sum+= r_f
	}
	fmt.Printf("Exp random average is: %.3fn", sum/100000)
	sum = 0.0
	logListToJson(list_f, "ExpRand.log")
	for i := 0; i < 100000; i++ {
		r_f := math.Log(1/rand.Float64())/0.23
		list_f[i] = r_f
		sum+= r_f
	}
	fmt.Printf("Ln 1/random average is: %.3fn", sum/100000)
	logListToJson(list_f, "Ln1Rand.log")
}

//export wrapperSimulate
func wrapperSimulate(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64,
		occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
		electrode_occupation []float64, site_energies []float64, hops int, record bool, traffic []float64, average_occupation []float64) float64 {
	//Log(fmt.Sprintf("%d %d", NSites, NElectrodes))
	newDistances := deFlattenFloatTo32(distances, NSites+NElectrodes, NSites+NElectrodes)
	newConstants := deFlattenFloatTo32(transitions_constant, NSites+NElectrodes, NSites+NElectrodes)
	bool_occupation := make([]bool, NSites)
	//printAverageExpRandom();
	time := simulate(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
		newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, false, record, traffic, average_occupation, 0)

	return time
}

//export wrapperSimulatePruned
func wrapperSimulatePruned(NSites int64, NElectrodes int64, prune_threshold float64, nu float64, kT float64, I_0 float64, R float64,
	occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
	electrode_occupation []float64, site_energies []float64, hops int, record bool, traffic []float64, 
	average_occupation []float64) float64 {
	newDistances := deFlattenFloatTo32(distances, NSites+NElectrodes, NSites+NElectrodes)
	newConstants := deFlattenFloatTo32(transitions_constant, NSites+NElectrodes, NSites+NElectrodes)
	bool_occupation := make([]bool, NSites)
	time := simulate(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
		newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, false, record, traffic, average_occupation, float32(prune_threshold))

	return time
}

//export wrapperSimulateCombined
func wrapperSimulateCombined(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64,
	occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
	electrode_occupation []float64, site_energies []float64, hops int, record bool, traffic []float64, average_occupation []float64) float64 {
//Log(fmt.Sprintf("%d %d", NSites, NElectrodes))
newDistances := deFlattenFloatTo32(distances, NSites+NElectrodes, NSites+NElectrodes)
newConstants := deFlattenFloatTo32(transitions_constant, NSites+NElectrodes, NSites+NElectrodes)
bool_occupation := make([]bool, NSites)
//printAverageExpRandom();
time := simulateCombined(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
	newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, traffic, average_occupation)

return time
}

//export wrapperSimulateRecord
func wrapperSimulateRecord(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, 
	occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
	electrode_occupation []float64, site_energies []float64, hops int, record bool, traffic []float64, average_occupation []float64) float64 {
	newDistances := deFlattenFloatTo32(distances, NSites+NElectrodes, NSites+NElectrodes)
	newConstants := deFlattenFloatTo32(transitions_constant, NSites+NElectrodes, NSites+NElectrodes)

	bool_occupation := make([]bool, NSites)
	for i := int64(0); i < NSites; i++{
		if false {//occupation[i] > 0{
			bool_occupation[i] = true
		} else {
			bool_occupation[i] = false
		}
	}
	time := simulate(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
	newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, true, record, traffic, average_occupation,
	0)

return time
}

//export analyzeStateOverlap
func analyzeStateOverlap(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, 
	occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
	electrode_occupation []float64, site_energies []float64, hops int, record bool, traffic []float64, average_occupation []float64) float64 {
	newDistances := deFlattenFloatTo32(distances, NSites+NElectrodes, NSites+NElectrodes)
	newConstants := deFlattenFloatTo32(transitions_constant, NSites+NElectrodes, NSites+NElectrodes)
	bool_occupation := make([]bool, NSites)
	for i := int64(0); i < NSites; i++{
		if occupation[i] > 0{
			bool_occupation[i] = true
		} else {
			bool_occupation[i] = false
		}
	}
	state_count := simulateReturnStatecount(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
		newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, true, record, traffic, average_occupation,
		0)
	fmt.Printf("Number of states in original: %d", len(state_count))
	for j := 0; j < 10; j++ {
		for i := int64(0); i < NSites; i++{
			if occupation[i] > 0{
				bool_occupation[i] = true
			} else {
				bool_occupation[i] = false
			}
		}
		other_state_count := simulateReturnStatecount(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
		newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, true, record, traffic, average_occupation,
		0)
		overlap := uint32(0)
		for key, val :=  range state_count {
			val2, ok := other_state_count[key]
			if ok {
				if val > val2 {
					overlap += uint32(val2)
				} else {
					overlap += uint32(val)
				}
			}
		}
		fmt.Printf("The number of keys in new try:%d\n%d: The overlap for %d hops was %d\n\n", len(other_state_count), j, hops, overlap)
	}


return 0
}

//export wrapperSimulateProbability
func wrapperSimulateProbability(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, 
	occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
	electrode_occupation []float64, site_energies []float64, hops int, record bool, traffic []float64, average_occupation []float64) float64 {

	newDistances := deFlattenFloat64(distances, NSites+NElectrodes, NSites+NElectrodes)
	newConstants := deFlattenFloat64(transitions_constant, NSites+NElectrodes, NSites+NElectrodes)
	
	for j := 0; j < int(NSites); j++ {
		occupation[j] = float64(0.5)
	}
	time := probSimulate(int(NSites), int(NElectrodes), nu, kT, I_0, R, occupation, 
			newDistances , E_constant, newConstants, electrode_occupation, site_energies, hops, record, traffic, average_occupation)

	return time
}

func simulateProbabilityReturnChannel(NSites int64, NElectrodes int64, nu float64, 
	kT float64, I_0 float64, R float64, occupation []float64, distances []float64, 
	E_constant []float64, transitions_constant []float64, electrode_occupation []float64, 
	site_energies []float64, hops int, record bool, c chan float64) {
	
	time := wrapperSimulateProbability(NSites, NElectrodes, nu, kT, I_0, R, occupation, 
		distances, E_constant, transitions_constant, electrode_occupation, site_energies,
		hops, false, nil, nil)
	c <- time
}

//export startProbabilitySimulation
func startProbabilitySimulation(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, 
	occupation []float64, distances []float64, E_constant []float64, transitions_constant []float64,
	electrode_occupation []float64, site_energies []float64, hops int, record bool) int64 {
	
	index := channelMapIndex
	channelMapIndex++
	c := make(chan float64)
	channelsMap[index] = c
	fmt.Printf("distance pointer: %p\n",&distances)

	fmt.Printf("pointer: %p\n",&E_constant)

	nE_constant := make([]float64, len(E_constant))
	copy(nE_constant, E_constant)
	fmt.Printf("pointer: %p\n",&nE_constant)

	go simulateProbabilityReturnChannel(NSites, NElectrodes, nu, kT, I_0, R, occupation, 
		distances, nE_constant, transitions_constant, electrode_occupation, site_energies,
		hops, record, c)

	return int64(index)
}

//export getResult
func getResult(index int64) float64 {
	result := <- channelsMap[int(index)]
	return result
}