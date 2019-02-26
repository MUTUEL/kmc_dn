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

func calcTransitions(transitions [][]float32, distances [][]float32, occupation []bool, 
    site_energies []float32, R float32, I_0 float32, kT float32, nu float32, NSites int, 
    N int, transitions_constant [][]float32) {
    for i := 0; i < N; i++ {
        for j := 0; j < N; j++ {
            if !transition_possible(i, j, NSites, occupation){
                transitions[i][j] = 0
            } else {
                var dE float32
                if i < NSites && j < NSites {
                    dE = site_energies[j] - site_energies[i] - I_0*R/distances[i][j]
                } else {
                    dE = site_energies[j] - site_energies[i]
                }
                if dE > 0 {
                    transitions[i][j] = nu * float32(math.Exp(float64(-dE/kT)))
                } else {
                    transitions[i][j] = 1
                }
                transitions[i][j]*=transitions_constant[i][j]
            }
        }
    }
}

func makeJump(occupation []bool, electrode_occupation []float64, site_energies []float32, 
    distances [][]float32, R float32, I_0 float32, NSites int, from int, to int) {
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

type probabilities struct {
    probList []float32
}




func simulate(NSites int, NElectrodes int, nu float32, kT float32, I_0 float32, R float32, time float64,
        occupation []bool, distances [][]float32, E_constant []float32, transitions_constant [][]float32,
        electrode_occupation []float64, site_energies []float32, hops int, record_problist bool, record bool, 
        traffic []float64, average_occupation []float64) float64 {
    N := NSites + NElectrodes
    transitions := make([][]float32, N)
    //occupation_time := make([]float64, NSites)
    allProbs := make(map[uint64]*probabilities)
    countProbs := make(map[uint64]uint16)

    //fmt.Printf("Site energies at start: %v\n", site_energies)
    for i := 0; i < NSites; i++ {
        acceptor_interaction := float32(0)
        for j := 0; j < NSites; j++ {
            if j != i && !occupation[j] {
                acceptor_interaction+= 1/distances[i][j]
            }
        }
        site_energies[i] = E_constant[i] - I_0*R*acceptor_interaction
    }

    for i := 0; i < NElectrodes; i++ {
		electrode_occupation[i] = 0.0
    }
    time = 0

    for i := 0; i < N; i++ {
        transitions[i] = make([]float32, NSites+NElectrodes)
    }
    countReuses := uint64(0)
    countStorage := uint64(0)
    reuseThreshold := uint16(1)
    reuseThresholdIncrease := uint64(100000)
    //showStep := 1
    for hop := 0; hop < hops; hop++ {
        var probList []float32
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
    
            probList = make([]float32, N*N)

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
                val, ok := countProbs[key64]
                if ok {

                    countProbs[key64]+=1
                    if val >= reuseThreshold {
                        newProbability := probabilities{probList}
                        allProbs[key64] = &newProbability
                        countStorage++
                        if countStorage > reuseThresholdIncrease {
                            reuseThreshold++
                            reuseThresholdIncrease+=100000
                        }
                    }
                } else {
                    countProbs[key64] = 1
                }
            }
        }
        time_step := rand.ExpFloat64() / float64(probList[len(probList)-1])
        time += time_step
        eventRand := rand.Float32() * probList[len(probList)-1]
        event := 0
        for i := 0; i < len(probList); i++ {
            if probList[i] >= eventRand {
                event = i
                break
            }
        }
        from := event/N
        to := event%N

        if record {
            traffic[event]+=1
            traffic[to*N+from]-=1
            for i := 0; i < NSites; i++ {
                if occupation[i] {
                    average_occupation[i]+=time_step
                }
            }
        }

        /*if hop % showStep == 0 {
            showStep*=2
            current := electrode_occupation[0] / time
            fmt.Printf("Hop: %d, current: %.3f, time: %.2f, reuse: %d, storage: %d\n", hop, current, time, countReuses, countStorage)
        }*/
        makeJump(occupation, electrode_occupation, site_energies, distances, R, I_0, 
            NSites, from, to)
        /*for i := 0; i < NSites; i++ {
            if occupation[i] {
                occupation_time[i]+=time_step
            }
        }*/
    }
    /*for i := 0; i < NSites; i++ {
        occupation_time[i]/=time
    }*/
    return time
}

type transition struct {
    from int;
    to int;
    rate float32;
    prunedList []*transition;
    count int;
}

/*type transitionBucket struct {
    transitions []transition;
    currentCount int;
}

func simulatePruned(NSites int, NElectrodes int, nu float64, kT float64, I_0 float64, R float64, time float64,
    occupation []bool, distances [][]float64, E_constant []float64, transitions_constant [][]float64,
    electrode_occupation []float64, site_energies []float64, hops int, record_problist bool) float64 {
    N := NSites + NElectrodes
    occupation_time := make([]float64, NSites)


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

    for i := 0; i < NElectrodes; i++ {
        electrode_occupation[i] = 0.0
    }
    time = 0
    number_of_buckets := 2

    buckets := make([]transitionBucket, 2)
    total_transitions := NSites*(NSites-1) + NSites * 2 * NElectrodes
    transitions_per_bucket := total_transitions / number_of_buckets +1
    for i := 0; i < number_of_buckets; i++ {
        buckets[i].transitions = make([]transition, transitions_per_bucket)
        buckets[i].currentCount = 0
    }

    for i:= 0; i < N; i++ {
        for j := 0; j < N; j++ {
            if i == j || (i >= NSites && j >= NSites) {
                continue
            }
            rnd := rand.Intn(number_of_buckets)
            while (buckets[rnd].currentCount >= transitions_per_bucket) {
                rnd = rand.Intn(number_of_buckets)
            }
            new_trans = transition{i, j, 0.0}
            buckets[rnd].transitions[buckets[rnd].currentCount] = new_trans
            buckets[rnd].currentCount++
        }
    }

    countReuses := uint64(0)
    countStorage := uint64(0)
    reuseThreshold := uint16(1)
    reuseThresholdIncrease := uint64(100000)
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
                val, ok := countProbs[key64]
                if ok {

                    countProbs[key64]+=1
                    if val >= reuseThreshold {
                        newProbability := probabilities{probList}
                        allProbs[key64] = &newProbability
                        countStorage++
                        if countStorage > reuseThresholdIncrease {
                            reuseThreshold++
                            reuseThresholdIncrease+=100000
                        }
                    }
                } else {
                    countProbs[key64] = 1
                }
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
            showStep*=2
            current := electrode_occupation[0] / time
            fmt.Printf("Hop: %d, current: %.3f, time: %.2f, reuse: %d, storage: %d\n", hop, current, time, countReuses, countStorage)
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
    return time
}*/



func main() {
    fmt.Println("hello world")
}
