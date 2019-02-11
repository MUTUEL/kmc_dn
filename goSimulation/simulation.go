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




func simulate(NSites int, NElectrodes int, nu float64, kT float64, I_0 float64, R float64, time float64,
        occupation []bool, distances [][]float64, E_constant []float64, transitions_constant [][]float64,
        electrode_occupation []float64, site_energies []float64, hops int, record_problist bool) float64 {
    N := NSites + NElectrodes
    transitions := make([][]float64, N)
    occupation_time := make([]float64, NSites)
    allProbs := make(map[uint64]*probabilities)

    //fmt.Printf("Site energies at start: %v\n", site_energies)
    for i := 0; i < NSites; i++ {
        acceptor_interaction := float64(0)
        for j := 0; j < NSites; j++ {
            if j != i && !occupation[j] {
                acceptor_interaction+= 1/distances[i][j]
            }
        }
        site_energies[i] = E_constant[i] - I_0*R*acceptor_interaction
    }
    fmt.Printf("Site energies after: %v", site_energies)

    for i := 0; i < NElectrodes; i++ {
		electrode_occupation[i] = 0.0
    }
    time = 0

    for i := 0; i < N; i++ {
        transitions[i] = make([]float64, NSites+NElectrodes)
    }
    countReuses := uint64(0)
    showStep := 1
    for hop := 0; hop < hops; hop++ {
        var probList []float64
        ok := false
        var key64 uint64
        if record_problist {
            var val *probabilities
            key64 = getKey(occupation)
            val, ok = allProbs[key64]
            if ok {
                probList = val.probList
                countReuses++
            }
        }
        if !ok {
            calcTransitions(transitions, distances, occupation, site_energies, R, I_0, kT, nu, NSites, N, transitions_constant)
    
            probList = make([]float64, N*N)

            for i := 0; i < N; i++ {
                for j := 0; j < N; j++ {
                    if i==0 && j==0 {
                        probList[0] = transitions[i][j]
                    } else {
                        probList[N*i+j] = probList[N*i+j-1] + transitions[i][j]
                    }
                }
            }
            if record_problist {
                newProbability := probabilities{probList}
                allProbs[key64] = &newProbability
            }
        }
        time_step := rand.ExpFloat64() / (probList[len(probList)-1])
        time += time_step
        eventRand := rand.Float64() * probList[len(probList)-1]
        event := 0
        for i := 0; i < len(probList); i++ {
            if probList[i] >= eventRand {
                event = i
                break
            }
        }
        from := event/N
        to := event%N

        if hop % showStep == 0 {
            showStep*=4
            current := electrode_occupation[0] / time
            fmt.Printf("Hop: %d, current: %.3f, time: %.2f\n", hop, current, time)
            /*fmt.Printf("Site energies at hop: %v\n", site_energies)
            fmt.Printf("Occupation at hop: %v\n", occupation)
            fmt.Printf("Transitions at hop: %.4v\n", transitions)
            fmt.Printf("Problist at hop: %.4v\n", probList)
            fmt.Printf("from:%d, to:%d\n", from, to)
            fmt.Printf("Nu:%.3f, kT:%.3f, trConst:%.4f", nu, kT, transitions_constant)
            for i := 0; i < len(transitions); i++ {
                for j := 0; j < len(transitions[i]); j++{
                    //fmt.Printf("i:%d, j:%d, val:%.4f\n", i, j, transitions[i][j])
                    var dE float64
                    if !transition_possible(i, j, NSites, occupation){
                        continue
                    }
                    if i < NSites && j < NSites {
                        dE = site_energies[j] - site_energies[i] - I_0*R/distances[i][j]
                    } else {
                        dE = site_energies[j] - site_energies[i]
                    }
                    if dE > 0 {
                        fmt.Printf("i:%d, j:%d, val:%.4f,dE:%.4f\n", i, j, nu * math.Exp(-dE/kT), dE)
                    } else {
                        fmt.Printf("i:%d, j:%d, val:%.4f,dE:%.4f\n", i, j, nu, dE)
                    }
                }
            }*/
        }
        makeJump(occupation, electrode_occupation, site_energies, distances, R, I_0, 
            NSites, from, to)
        for i := 0; i < NSites; i++ {
            if occupation[i] {
                occupation_time[i]+=time_step
            }
        }
    }
    for i := 0; i < NSites; i++ {
        occupation_time[i]/=time
    }
    fmt.Printf("Time: %.2f\n", time)
    fmt.Printf("Occupation percentage: %.3v", occupation_time)

    fmt.Println(electrode_occupation)
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

func calcTransitions(transitions [][]float64, distances [][]float64, occupation []bool, 
    site_energies []float64, R float64, I_0 float64, kT float64, nu float64, NSites int, 
    N int, transitions_constant [][]float64) {
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
                    transitions[i][j] = nu * math.Exp(-dE/kT)
                } else {
                    transitions[i][j] = 1
                }
                transitions[i][j]*=transitions_constant[i][j]
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

func main() {
    fmt.Println("hello world")
}
