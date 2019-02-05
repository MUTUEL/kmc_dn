package main

import "fmt"
import "math"
import "math/rand"
//import "sort"

func wrapReturnables(time float64, occupation []bool, electrode_occupation []int) []float64 {
    returnable := make([]float64, 1 + len(occupation) + len(electrode_occupation))
    returnable[0] = time
    lo := len(occupation)
    for i := 0; i < lo; i++ {
        if occupation[i] {
            returnable[1+i] = 1.0
        } else {
            returnable[1+i] = 0.0
        }
    }
    for i := 0; i < len(electrode_occupation); i++ {
        returnable[1+lo+i] = float64(electrode_occupation[i])
    }
    return returnable
}

func getKey(boolAr []bool) uint64 {
    var r uint64 = 0
    for i := 0; i < len(boolAr); i++ {
        r = r<<1
        if boolAr[i] {
            r+=1
        }
    }
    return r
}

type probabilities struct {
    probList []float64
}

func calcTransitions(transitions [][]float64, distances [][]float64, occupation []bool, 
    site_energies []float64, R float64, I_0 float64, kT float64, nu float64, NSites int, 
    N int) {
    for i := 0; i < N; i++ {
        for j := 0; j < N; j++ {
            if !transition_possible(i, j, NSites, occupation){
                transitions[i][j] = 0
            } else {
                var dE float64
                if i < NSites && j < NSites {
                    dE = site_energies[j] - site_energies[i] - I_0*R/distances[i][j]
                } else {
                    dE = site_energies[j] - site_energies[i]
                }
                if dE > 0 {
                    transitions[i][j] = nu * math.Exp(float64(-dE/kT))
                } else {
                    transitions[i][j] = 1
                }
            }
        }
    }
}

func makeJump(occupation []bool, electrode_occupation []float64, site_energies []float64, 
    distances [][]float64, R float64, I_0 float64, NSites int, from int, to int) {
    if from < NSites {
        occupation[from] = false
        for j := 0; j < NSites; j++ {
            if j != from {
                site_energies[j] -= I_0*R*(1/distances[j][from])
            }
            
        }
    } else {
        electrode_occupation[from-NSites]-=1.0
    }
    if to < NSites {
        occupation[to] = true
        for j := 0; j < NSites; j++ {
            if j != to {
                site_energies[j] += I_0*R*(1/distances[j][to])
            }
        }
    } else {
        electrode_occupation[to-NSites]+=1.0
    }
}


func simulate(NSites int, NElectrodes int, nu float64, kT float64, I_0 float64, R float64, time float64,
        occupation []bool, distances [][]float64, E_constant []float64, transitions_constant float64,
        electrode_occupation []float64, site_energies []float64, hops int, record_problist bool) float64 {
    N := NSites + NElectrodes
    transitions := make([][]float64, N)
    allProbs := make(map[uint64]*probabilities)
    
    for i := 0; i < NSites; i++ {
        acceptor_interaction := float64(0)
        for j := 0; j < NSites; j++ {
            if j != i && !occupation[j] {
                acceptor_interaction+= 1/distances[i][j]
            }
        }
        site_energies[i] = E_constant[i] - I_0*R*acceptor_interaction
    }

    for i := 0; i < N; i++ {
        transitions[i] = make([]float64, NSites+NElectrodes)
    }
    countReuses := uint64(0)
    propabilityMedianCounter := 0
    countMedians := 0
    randStat := 0
    showStep := 1
    for hop := 0; hop < hops; hop++ {
        var probList []float64
        key64 := getKey(occupation)
        if val, ok := allProbs[key64]; ok {
            probList = val.probList
            countReuses++
        } else {
            calcTransitions(transitions, distances, occupation, site_energies, R, I_0, kT, nu, NSites, N)
            probList = make([]float64, N*N)
            //testList := make([]float64, N*N)

            for i := 0; i < N; i++ {
                for j := 0; j < N; j++ {
                    //testList[N*i+j] = transitions[i][j]
                    if i==0 && j==0 {
                        probList[0] = transitions[i][j]
                    } else {
                        probList[N*i+j] = probList[N*i+j-1] + transitions[i][j]
                    }
                }
            }

            /*sort.Float64s(testList)
            summa := float64(0)
            for i := 0; i < len(testList); i++ {
                summa += testList[i]
                if summa >= probList[len(probList)-1]/100.0 {
                    countMedians++
                    propabilityMedianCounter+=i
                    break
                }
            }*/
            if record_problist {
                newProbability := probabilities{probList}
                allProbs[key64] = &newProbability
            }
        }
        time += rand.ExpFloat64() / (probList[len(probList)-1] * transitions_constant) //TODO this is a potential error
        eventRand := rand.Float64() * probList[len(probList)-1]
        event := 0
        for i := 0; i < len(probList); i++ {
            if probList[i] >= eventRand {
                event = i
                randStat += i
                break
            } 
        }
        from := event/N
        to := event%N
        makeJump(occupation, electrode_occupation, site_energies, distances, R, I_0, 
            NSites, from, to)
        if hop % showStep == 0 {
            showStep*=2
            current := electrode_occupation[0] / time
            fmt.Println("Hop: %d, current: %.2f", hop, current)
        }
    }
    fmt.Println(time)
    fmt.Println(site_energies)
    fmt.Println("Propability list was reused %d times\n", countReuses)
    fmt.Println("The average of propability median is %.2f, number of medians counted %d\n", float64(propabilityMedianCounter)/float64(countMedians), countMedians)
    fmt.Println("The average of randStat ", float64(randStat)/float64(hops))

    fmt.Println(electrode_occupation)
    //return wrapReturnables(time, occupation, electrode_occupation)
    return time
}

func transition_possible(i int, j int, NSites int, occupation []bool) bool {
    if i >= NSites && j >= NSites {
        return false
    } else if i>= NSites && occupation[j]{
        return false
    } else if j>= NSites && !occupation[i] {
        return false
    }
    if i < NSites && j < NSites && (!occupation[i] || occupation[j]){
        return false
    }
    return true
}

func main() {
    fmt.Println("hello world")
}
