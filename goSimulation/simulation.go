package main

import (
    "fmt"
    "math"
    "math/rand"
    "io/ioutil" 
    "encoding/json"
    )
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
    if i == j {
        return false
    }
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


func calcTransitionList(transitions []transition, distances [][]float32, occupation []bool, 
    site_energies []float32, R float32, I_0 float32, kT float32, nu float32, NSites int, 
    N int, transitions_constant [][]float32) {

    for i, trans := range transitions {
        if !transition_possible(trans.from, trans.to, NSites, occupation){
            transitions[i].rate = 0
        } else {
            var dE float32
            if trans.from < NSites && trans.to < NSites {
                dE = site_energies[trans.to] - site_energies[trans.from] - I_0*R/distances[trans.from][trans.to]
            } else {
                dE = site_energies[trans.to] - site_energies[trans.from]
            }
            if dE > 0 {
                transitions[i].rate = nu * float32(math.Exp(float64(-dE/kT)))
            } else {
                transitions[i].rate = 1
            }
            transitions[i].rate*=transitions_constant[trans.from][trans.to]
        }
    }
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

type transition struct {
    from int;
    to int;
    rate float32;
}

type reuseLog struct {
    Reuses []uint64
    States []int
}

func logReuseToJson(reuses []uint64, states []int, file_name string){
    var log_list []reuseLog
    data, err := ioutil.ReadFile(file_name)
    if err == nil {
        err2 := json.Unmarshal(data, &log_list)
        if err2 != nil {
            fmt.Printf("opening config file %s", err2.Error())
        }
    }

    log := reuseLog {reuses, states}
    log_list = append(log_list, log)
    logjson, _ := json.Marshal(log_list)
    _ = ioutil.WriteFile(file_name, logjson, 0644)
}

func getRandomEvent(probList []float32) int {
    eventRand := rand.Float32() * probList[len(probList)-1]
    event := 0
    e_step := int(len(probList)/2)
    i := e_step
    for true {
        if e_step >= 2 {
            e_step = e_step/2
        }
        if probList[i] < eventRand {
            i+=e_step
            if i >= len(probList) {
                i = len(probList) - 1
            }
        } else if (i>0 && probList[i-1] >= eventRand) {
            i-=e_step
            if i < 0 {
                i = 0
            }
        } else {
            event = i
            break
        }
    }
    return event
}


  


func simulate(NSites int, NElectrodes int, nu float32, kT float32, I_0 float32, R float32,
        occupation []bool, distances [][]float32, E_constant []float32, transitions_constant [][]float32,
        electrode_occupation []float64, site_energies []float32, hops int, record_problist bool, record bool, 
        traffic []float64, average_occupation []float64, transition_cut_constant float32) float64 {
    N := NSites + NElectrodes
    transitions := make([]transition, 0, N*N)
    largest_tc := float32(0)
    for i := 0; i < len(transitions_constant); i++ {
        for j := 0; j < len(transitions_constant[i]); j++ {
            if largest_tc < transitions_constant[i][j] {
                largest_tc = transitions_constant[i][j]
            }
        }
    }

    for i := 0; i < len(transitions_constant); i++ {
        for j := 0; j < len(transitions_constant[i]); j++ {
            if transitions_constant[i][j] > (transition_cut_constant*largest_tc) {
                transitions = append(transitions, transition{i, j, 0})
            }
        }
    }
    if transition_cut_constant > 0 {
        //fmt.Printf("Transition list size: %d", len(transitions))
    }
    

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
    time := float64(0)

    countReuses := uint64(0)
    countStorage := uint64(0)
    reuseThreshold := uint16(1)
    reuseThresholdIncrease := uint64(100000)
    /*countResighting := uint64(0)
    reuses := make([]uint64, 30)
	states := make([]int, 30)
    showStep := 2
    showIndex := 0*/
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
                //countResighting++
            }
        }
        if !ok {
            calcTransitionList(transitions, distances, occupation, site_energies, R, I_0, kT, nu, NSites, N, transitions_constant)
    
            probList = make([]float32, len(transitions))


            for i,trans := range transitions {
                if i == 0 {
                    probList[0] = trans.rate
                } else {
                    probList[i] = probList[i-1] + trans.rate
                }
            }

            if record_problist {
                val, ok := countProbs[key64]
                if ok {
                    //countResighting++
                    countProbs[key64]++
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
        event := getRandomEvent(probList)
        /*if hop % showStep == showStep - 1{
            showStep*=2
            reuses[showIndex] = countResighting
            states[showIndex] = len(countProbs)
            showIndex+=1
        }*/
        from := transitions[event].from
        to := transitions[event].to

        if record {
            traffic[from*N+to]+=1
            traffic[to*N+from]-=1
            for i := 0; i < NSites; i++ {
                if occupation[i] {
                    average_occupation[i]+=time_step
                }
            }
        }
        makeJump(occupation, electrode_occupation, site_energies, distances, R, I_0, 
            NSites, from, to)
    }

    //logReuseToJson(reuses, states, "reusing.log")

    return time
}

func simulateRecordPlus(NSites int, NElectrodes int, nu float32, kT float32, I_0 float32, R float32,
    occupation []bool, distances [][]float32, E_constant []float32, transitions_constant [][]float32,
    electrode_occupation []float64, site_energies []float32, hops int, record_problist bool, record bool, 
    traffic []float64, average_occupation []float64, transition_cut_constant float32) float64 {
    N := NSites + NElectrodes
    transitions := make([]transition, 0, N*N)
    largest_tc := float32(0)
    for i := 0; i < len(transitions_constant); i++ {
        for j := 0; j < len(transitions_constant[i]); j++ {
            if largest_tc < transitions_constant[i][j] {
                largest_tc = transitions_constant[i][j]
            }
        }
    }

    for i := 0; i < len(transitions_constant); i++ {
        for j := 0; j < len(transitions_constant[i]); j++ {
            if transitions_constant[i][j] > (transition_cut_constant*largest_tc) {
                transitions = append(transitions, transition{i, j, 0})
            }
        }
    }
    if transition_cut_constant > 0 {
        //fmt.Printf("Transition list size: %d", len(transitions))
    }


    //occupation_time := make([]float64, NSites)
    allProbs := make(map[uint64]*probabilities)
    countProbs := make(map[uint64]uint16)

    //fmt.Printf("Site energies at start: %v\n", site_energies)


    for i := 0; i < NElectrodes; i++ {
        electrode_occupation[i] = 0.0
    }
    time := float64(0)

    countReuses := uint64(0)
    countStorage := uint64(0)
    reuseThreshold := uint16(1)
    reuseThresholdIncrease := uint64(100000)
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
                //countResighting++
            }
        }
        if !ok {
            for i := 0; i < NSites; i++ {
                acceptor_interaction := float32(0)
                for j := 0; j < NSites; j++ {
                    if j != i && !occupation[j] {
                        acceptor_interaction+= 1/distances[i][j]
                    }
                }
                site_energies[i] = E_constant[i] - I_0*R*acceptor_interaction
            }
            calcTransitionList(transitions, distances, occupation, site_energies, R, I_0, kT, nu, NSites, N, transitions_constant)

            probList = make([]float32, len(transitions))


            for i,trans := range transitions {
                if i == 0 {
                    probList[0] = trans.rate
                } else {
                    probList[i] = probList[i-1] + trans.rate
                }
            }

            if record_problist {
                val, ok := countProbs[key64]
                if ok {
                    //countResighting++
                    countProbs[key64]++
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
        event := getRandomEvent(probList)

        from := transitions[event].from
        to := transitions[event].to
        if from < NSites {
            occupation[from] = false
        } else {
            electrode_occupation[from-NSites]-=1.0
        }
        if to < NSites {
            occupation[to] = true
        } else {
            electrode_occupation[to-NSites]+=1.0
        }
    }
    return time
}

func simulateReturnStatecount(NSites int, NElectrodes int, nu float32, kT float32, I_0 float32, R float32,
    occupation []bool, distances [][]float32, E_constant []float32, transitions_constant [][]float32,
    electrode_occupation []float64, site_energies []float32, hops int, record_problist bool, record bool, 
    traffic []float64, average_occupation []float64, transition_cut_constant float32) map[uint64]uint32 {
    N := NSites + NElectrodes
    transitions := make([]transition, 0, N*N)
    largest_tc := float32(0)
    for i := 0; i < len(transitions_constant); i++ {
        for j := 0; j < len(transitions_constant[i]); j++ {
            if largest_tc < transitions_constant[i][j] {
                largest_tc = transitions_constant[i][j]
            }
        }
    }

    for i := 0; i < len(transitions_constant); i++ {
        for j := 0; j < len(transitions_constant[i]); j++ {
            if transitions_constant[i][j] > (transition_cut_constant*largest_tc) {
                transitions = append(transitions, transition{i, j, 0})
            }
        }
    }
    if transition_cut_constant > 0 {
        //fmt.Printf("Transition list size: %d", len(transitions))
    }


    //occupation_time := make([]float64, NSites)
    allProbs := make(map[uint64]*probabilities)
    countProbs := make(map[uint64]uint32)

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
    time := float64(0)

    countReuses := uint64(0)
    countStorage := uint64(0)
    reuseThreshold := uint32(1)
    reuseThresholdIncrease := uint64(100000)
    /*countResighting := uint64(0)
    reuses := make([]uint64, 30)
    states := make([]int, 30)
    showStep := 2
    showIndex := 0*/
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
                countProbs[key64]++
                //countResighting++
            }
        }
        if !ok {
            calcTransitionList(transitions, distances, occupation, site_energies, R, I_0, kT, nu, NSites, N, transitions_constant)

            probList = make([]float32, len(transitions))


            for i,trans := range transitions {
                if i == 0 {
                    probList[0] = trans.rate
                } else {
                    probList[i] = probList[i-1] + trans.rate
                }
            }

            if record_problist {
                val, ok := countProbs[key64]
                if ok {
                    //countResighting++
                    countProbs[key64]++
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
        e_step := int(len(probList)/2)
        i := e_step
        for true {
            if e_step >= 2 {
                e_step = e_step/2
            }
            if probList[i] < eventRand {
                i+=e_step
                if i >= len(probList) {
                    i = len(probList) - 1
                }
            } else if (i>0 && probList[i-1] >= eventRand) {
                i-=e_step
                if i < 0 {
                    i = 0
                }
            } else {
                event = i
                break
            }
        }
        /*if hop % showStep == showStep - 1{
            showStep*=2
            reuses[showIndex] = countResighting
            states[showIndex] = len(countProbs)
            showIndex+=1
        }*/
        from := transitions[event].from
        to := transitions[event].to

        if record {
            traffic[from*N+to]+=1
            traffic[to*N+from]-=1
            for i := 0; i < NSites; i++ {
                if occupation[i] {
                    average_occupation[i]+=time_step
                }
            }
        }

        makeJump(occupation, electrode_occupation, site_energies, distances, R, I_0, 
            NSites, from, to)

    }

    //logReuseToJson(reuses, states, "reusing.log")

    return countProbs
}

func simulateCombined(NSites int, NElectrodes int, nu float32, kT float32, I_0 float32, R float32,
    occupation []bool, distances [][]float32, E_constant []float32, transitions_constant [][]float32,
    electrode_occupation []float64, site_energies []float32, hops int, traffic []float64, average_occupation []float64) float64 {
    N := NSites + NElectrodes
    transitions := make([][]float32, N)
    //occupation_time := make([]float64, NSites)

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
    time := float64(0)

    for i := 0; i < N; i++ {
        transitions[i] = make([]float32, NSites+NElectrodes)
    }
    
    var probList []float32
    probList = make([]float32, N*N)
    
    for hop := 0; hop < hops; hop++ {
        calcTransitions(transitions, distances, occupation, site_energies, R, I_0, kT, nu, NSites, N, transitions_constant)

        for i := 0; i < N; i++ {
            for j := 0; j < N; j++ {
                if i==0 && j==0 {
                    probList[0] = transitions[i][j]
                } else {
                    probList[N*i+j] = probList[N*i+j-1] + transitions[i][j]
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

        traffic[event]+=1
        traffic[to*N+from]-=1
        for i := 0; i < NSites; i++ {
            if occupation[i] {
                average_occupation[i]+=time_step
            }
        }

        makeJump(occupation, electrode_occupation, site_energies, distances, R, I_0, 
            NSites, from, to)
    }

    ave_occupation := make([]float64, len(occupation))
    for i:= 0; i < len(occupation); i++ {
        ave_occupation[i] = average_occupation[i]/time
    }

    electrode_currents := calcAverageCurrent(NSites, NElectrodes, nu, kT, I_0, R, 
        time, ave_occupation, distances, E_constant, transitions_constant, site_energies)
    for i:= 0; i < NElectrodes; i++ {
        electrode_occupation[i] = electrode_currents[i]*time
    }

    return time
}

func calcAverageCurrent(NSites int, NElectrodes int, nu float32, kT float32, I_0 float32, 
        R float32, time float64, occupation []float64, distances [][]float32, 
        E_constant []float32, transitions_constant [][]float32, site_energies []float32) []float64 {
    N := NSites + NElectrodes
    transitions := make([][]float64, N)
    for i := 0; i < N; i++{
        transitions[i] = make([]float64, N)
    }
    eoDifference := make([]float64, NElectrodes)

    for i := 0; i < NSites; i++ {
        acceptor_interaction := float64(0)
        for j := 0; j < NSites; j++ {
            if j != i {
                acceptor_interaction+= (1-occupation[j])/float64(distances[i][j])
            }
        }
        site_energies[i] = E_constant[i] - I_0*R*float32(acceptor_interaction)
    }
    for i:= 0; i < NElectrodes; i++ {
        eoDifference[i] = 0
    }

	tot_rates := 0.0
    for i := 0; i < N; i++ {
        for j := 0; j < N; j++ {
            base_prob := probTransitionPossible(i, j, NSites, occupation)
			var dE float32
			if i < NSites && j < NSites {
				dE = site_energies[j] - site_energies[i] - I_0*R/distances[i][j]
			} else {
				dE = site_energies[j] - site_energies[i]
			}
			
			if dE > 0 {
				transitions[i][j] = base_prob * float64(nu) * math.Exp(float64(-dE/kT))
			} else {
				transitions[i][j] = base_prob * float64(nu)
			}
			transitions[i][j]*=float64(transitions_constant[i][j])
			tot_rates+=transitions[i][j]
        }
    }
    time_step := 0.98 / tot_rates
    for i := 0; i < N; i++ {
        for j := 0; j < N; j++ {
            if i >= NSites && j >= NSites {
                break
            }
            rate := float64(transitions[i][j]/tot_rates)
            if i >= NSites {
                eoDifference[i-NSites] -= rate
            }
            if j >= NSites {
                eoDifference[j-NSites]+= rate
            }
        }
    }
    electrode_currents := make([]float64, NElectrodes)
    for i:= 0; i < NElectrodes; i++ {
        electrode_currents[i] = eoDifference[i]/time_step
    }
    return electrode_currents
}



func main() {
    fmt.Println("hello world")
}
