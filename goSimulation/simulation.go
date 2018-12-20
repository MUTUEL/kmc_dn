package main

import "fmt"
import "math"
import "math/rand"


func simulate(NSites int, NElectrodes int, nu float64, kT float64, I_0 float64, R float64, time float64,
        occupation []bool, distances [][]float64, E_constant []float64, transitions_constant float64,
        electrode_occupation []int, site_energies []float64, hops int) float64 {
    N := NSites + NElectrodes
    transitions := make([][]float64, N)
    probList := make([]float64, N*N)
    Log(fmt.Sprintf("%v", site_energies))
    for i := 0; i < NSites; i++ {
        acceptor_interaction := float64(0)
        for j := 0; j < NSites; j++ {
            if j != i && !occupation[j] {
                acceptor_interaction+= 1/distances[i][j]
            }
        }
        site_energies[i] = E_constant[i] - I_0*R*acceptor_interaction
    }
    Log(fmt.Sprintf("%v", site_energies))

    for i := 0; i < N; i++ {
        transitions[i] = make([]float64, NSites+NElectrodes)
    }
    for hop := 0; hop < hops; hop++ {
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

        for i := 0; i < N; i++ {
            for j := 0; j < N; j++ {
                if i==0 && j==0 {
                    probList[0] = transitions[i][j]
                } else {
                    probList[N*i+j] = probList[N*i+j-1] + transitions[i][j]
                }
            }
        }
        time += rand.ExpFloat64() / (probList[len(probList)-1] * transitions_constant) //TODO this is a potential error
        eventRand := rand.Float64() * probList[len(probList)-1]
        minim := 0
        maxim := len(probList) -1
        event := maxim/2
        for minim != maxim {
            if probList[event] > eventRand {
                maxim = event - 1
            } else {
                minim = event
            }
            event = (minim+maxim) / 2 + 1
        }
        ii := event/N
        ij := event%N
        if ii < NSites {
            occupation[ii] = false
            for j := 0; j < NSites; j++ {
                site_energies[j] -= I_0*R*(1/distances[j][ii])
            }
        } else {
            electrode_occupation[ii-NSites]-=1
        }
        if ij < NSites {
            occupation[ij] = true
            for j := 0; j < NSites; j++ {
                site_energies[j] += I_0*R*(1/distances[j][ij])
            }
        } else {
            electrode_occupation[ij-NSites]+=1
        }
    }
    fmt.Println(time)
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

func main() {
    fmt.Println("hello world")
}
