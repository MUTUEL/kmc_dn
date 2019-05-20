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

//export wrapperSimulateRecordPlus
func wrapperSimulateRecordPlus(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, 
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
		time := simulateRecordPlus(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
		newDistances , toFloat32(E_constant), newConstants, electrode_occupation, toFloat32(site_energies), hops, true, false, traffic, average_occupation,
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

func channelSimulateRecord(NSites int64, NElectrodes int64, nu float64, kT float64, I_0 float64, R float64, 
	occupation []float64, distances [][]float32, E_constant []float64, transitions_constant [][]float32,
	site_energies []float64, electrode_occupation []float64, hops int, c chan float64) {
		bool_occupation := make([]bool, NSites)
		for i := int64(0); i < NSites; i++{
			if occupation[i] > 0{
				bool_occupation[i] = true
			} else {
				bool_occupation[i] = false
			}
		}
		time := simulateRecordPlus(int(NSites), int(NElectrodes), float32(nu), float32(kT), float32(I_0), float32(R), bool_occupation, 
		distances , toFloat32(E_constant), transitions_constant, electrode_occupation, toFloat32(site_energies), hops, true, false, nil, nil,
		0)
		c <- time
}

//export parallelSimulations
func parallelSimulations(NSites []float64, NElectrodes []float64, nu []float64, 
		kT []float64, I_0 []float64, R []float64, occupation []float64, distances []float64, 
		E_constant []float64, transitions_constant []float64, electrode_occupation []float64, 
		hops []float64, time []float64, site_energies []float64) int64{

	channelsMap := make(map[int](chan float64))
	electrode_occupations := make(map[int]([]float64))
	totalSites := 0
	totalElectrodes := 0
	totalCombos := 0
	for i := 0; i < len(NSites); i++ {
		c := make(chan float64)
		channelsMap[i] = c
		newNSite := int(NSites[i])
		newElectrodes := int(NElectrodes[i])
		N := newNSite + newElectrodes
		newNu := nu[i]
		newKT := kT[i]
		newI_0 := I_0[i]
		newR := R[i]
		newHops := int(hops[i])
		newOccupation := occupation[totalSites:(totalSites+newNSite)]
		newDistances := deFlattenFloatTo32(distances[totalCombos:(totalCombos+N*N)], int64(N), int64(N))
		newE_constant := E_constant[totalSites:(totalSites+newNSite)]
		newTransitions_constant := deFlattenFloatTo32(transitions_constant[totalCombos:(totalCombos+N*N)], int64(N), int64(N))
		newSite_energies := site_energies[(totalElectrodes+totalSites):(totalElectrodes+totalSites+newNSite+newElectrodes)]
		newElectrode_occupation := electrode_occupation[totalElectrodes:(totalElectrodes+newElectrodes)]
		electrode_occupations[i] = newElectrode_occupation
		go channelSimulateRecord(int64(newNSite), int64(newElectrodes), newNu, newKT, newI_0, newR, 
			newOccupation, newDistances, newE_constant, newTransitions_constant, newSite_energies,
			newElectrode_occupation, newHops, c)
		totalSites+=newNSite
		totalElectrodes+=newElectrodes
		totalCombos+=N*N
	}
	totalElectrodes = 0
	for i := 0; i < len(NSites); i++ {
		time_result := <- channelsMap[i]
		time[i] = time_result
		for j := 0; j < int(NElectrodes[i]); j++ {
			if electrode_occupation[totalElectrodes+j] != electrode_occupations[i][j] {
				fmt.Println("NECESSARY 1")
			}
			electrode_occupation[totalElectrodes+j] = electrode_occupations[i][j]
		}

		totalElectrodes += int(NElectrodes[i])
	}
	return int64(0)
}