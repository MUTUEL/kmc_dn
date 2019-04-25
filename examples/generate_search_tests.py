import os
import sys
#sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import dn_search
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import kmc_dopant_networks_utils as kmc_utils
import dn_search_util
import matplotlib.pyplot as plt
import pickle
from validate_tests import compareVisualizeErrorDistribution


def generate_sample_test(N_acceptors, N_donors, electrode_placements, N_tests, fileName):
    electrodes = np.zeros((len(electrode_placements), 4))
    for i in range(len(electrode_placements)):
        for j in range(3):
            electrodes[i][j] = electrode_placements[i][j]
    kmc = kmc_dn.kmc_dn(N_acceptors, N_donors, 1, 1, 0, electrodes = electrodes)
    tests = []
    for t in range(N_tests):
        volts = []
        expected_currents = []
        for i in range(len(electrode_placements)-1):
            volt = random.random()-0.5
            volt = math.fabs(volt)/volt*volt*volt*320
            kmc.electrodes[i][3]=volt
            volts.append(volt)
        kmc.electrodes[len(electrode_placements)-1][3]=0
        kmc.update_V()
        volts.append(0)
        kmc.go_simulation(hops=100000, record=True)
        for i in range(len(kmc.current)):
            expected_currents.append((i, kmc.current[i]))
        if len(expected_currents) > 0:
            tests.append((volts, expected_currents))
            print ("test %d: %s"%(t, str(tests[t])))
    

    plt.clf()
    kmc_utils.visualize_traffic(kmc, 111, "Example network")
    plt.savefig(fileName)

    return tests

def get8Electrodes(xdim, ydim):
    electrodes = np.zeros((8, 4))
    electrodes[0] = [0, ydim/4, 0, 10]
    electrodes[1] = [0, 3*ydim/4, 0, 0]
    electrodes[2] = [xdim, ydim/4, 0, 10]
    electrodes[3] = [xdim, 3*ydim/4, 0, 0]
    electrodes[4] = [xdim/4, 0, 0, 10]
    electrodes[5] = [3*xdim/4, 0, 0, 0]
    electrodes[6] = [xdim/4, ydim, 0, 10]
    electrodes[7] = [3*xdim/4, ydim, 0, 0]

    return electrodes
        
def genAndSaveTest(fileName, N_acceptors, N_donors, N_tests):
    electrodes = get8Electrodes(1, 1)

    tests = generate_sample_test(N_acceptors, N_donors, electrodes, N_tests, "%s.png"%(fileName))
    with open("%s.kmc"%(fileName), "wb") as f:
        pickle.dump(tests, f)

def getRandomDn(N_acceptors, N_donors):
    electrodes = get8Electrodes(1, 1)
    dn = kmc_dn.kmc_dn(N_acceptors, N_donors, 1, 1, 0, electrodes = electrodes)
    return dn

def genXorTest(fileName, exampleFileName):
    dn = getRandomDn(10, 1)
    tests = []
    for i in range(100):
        rel_path = "tests/xor/test%d.kmc"%(i)
        abs_file_path = os.path.join(os.path.dirname(__file__), rel_path)
        dn.loadSelf(abs_file_path)
        volts = []
        currents = []
        for i in range(len(dn.electrodes)):
            volts.append(dn.electrodes[i][3])
            currents.append((i, dn.expected_current[i]))
        tests.append((volts, currents))
    dn.python_simulation(hops=1000000, record=True)
    plt.clf()
    kmc_utils.visualize_traffic(dn, 111, "Example network")
    plt.savefig(exampleFileName)
    with open("%s.kmc"%(fileName), "wb") as f:
        pickle.dump(tests, f)

def get_schedule1(multiplier, time_multiplier):
    return [
            (0.002*multiplier, 600*time_multiplier, 0),
            (0.001*multiplier, 1200*time_multiplier, 0),
            (0.001*multiplier, 1205*time_multiplier, 1),
            (0.0015*multiplier, 1800*time_multiplier, 1),
            (0.0005*multiplier, 2400*time_multiplier, 1),
            (0.0002*multiplier, 3000*time_multiplier, 1),
            (0.0001*multiplier, 3300*time_multiplier, 1),
            (0, 3600*time_multiplier, 2),
        ]

def get_schedule2(multiplier, time_multiplier):
    return [
        (0.2*multiplier, 1500*time_multiplier, 0),
        (0, 1505*time_multiplier, 1),
        (0.1*multiplier, 2500*time_multiplier, 1),
        (0, 2700*time_multiplier, 1)
    ]

def searchAnnealingBasedOnTest(fileName, N_acceptors, N_donors, test_index, schedule_function, use_tests, hours = 10, error_threshold_multiplier = 1):
    with open(fileName, 'rb') as f:
        tests = pickle.load(f)
        tests = orderTests(tests)
        dn = getRandomDn(N_acceptors, N_donors)
        search = dn_search.dn_search(dn, tests, 1, 1, 0.04, 0.04)
        schedule = schedule_function(error_threshold_multiplier, hours)
        search.setUseTests(use_tests)
        return search.simulatedAnnealingSearch(0.4, schedule, "10DOP%d"%(test_index))
    return None

def searchGeneticBasedOnTest(fileName, N_acceptors, N_donors, test_index, use_tests, 
        hours = 10, uniqueness = 5000, disparity=2, use_single_point_crossover = True,
        mut_pow=1, order_center = None, gen_size = 50):
    with open(fileName, 'rb') as f:
        tests = pickle.load(f)
        tests = orderTests(tests)
        dn = getRandomDn(N_acceptors, N_donors)
        search = dn_search.dn_search(dn, tests, 1, 1, 0.04, 0.04)
        search.setUseTests(use_tests)
        if use_single_point_crossover:
            cross_over_function = search.singlePointCrossover
        else:
            cross_over_function = search.alteredTwoPointCrossOver
        return search.genetic_search(gen_size, 3600*hours, 2, uniqueness, "10DOP%d"%(test_index), 
            cross_over_function = cross_over_function, mut_pow=mut_pow, order_center=order_center)
    return None

def searchRandomAndPlot(fileName, N_acceptors, N_donors, hours, test_index, use_tests):
    with open(fileName, 'rb') as f:
        tests = pickle.load(f)
        dn = getRandomDn(N_acceptors, N_donors)
        search = dn_search.dn_search(dn, tests, 1, 1, 0.04, 0.04)
        search.setUseTests(use_tests)

        bestDn, errors, vals, diffs = search.randomSearch(3600*hours)
        dict = {"errors":errors, "validations":vals, "differences":diffs}
        compareVisualizeErrorDistribution(dict, "randomSearch%d.png"%(test_index))

def orderTests(tests):
    ordered_tests = []
    remaining_tests = []
    best_distances = []
    for test in tests:
        remaining_tests.append(test)
    for i in range(len(tests)):
        best = remaining_tests[0]
        best_distance = testDistance(best, ordered_tests)
        for test in remaining_tests:
            distance = testDistance(test, ordered_tests)
            if distance > best_distance:
                best = test
                best_distance = distance
        ordered_tests.append(best)
        remaining_tests.remove(best)
        best_distances.append(best_distance)
    return ordered_tests

def testDistance(test, tests):
    if len(tests) == 0:
        return 0
    else:
        closest = nDimensionDistance(test[0], tests[0][0])
        for comp_test in tests:
            dist = nDimensionDistance(test[0], comp_test[0])
            if dist < closest: closest = dist
    return closest

def nDimensionDistance(a, b):
    sum = 0
    for i in range(len(a)):
        sum+= (a[i]-b[i])**2
    return sum

N_acceptors = 10
N_donors = 3
rel_path = "search_tests/test_set_5"
abs_file_path = os.path.join(os.path.dirname(__file__), rel_path)
#genXorTest(abs_file_path, "xor_example.png")

#genAndSaveTest(abs_file_path, N_acceptors, N_donors, 100)
tests = 50
hours = 3
index = 1
center = (0.5, 0.5, 0)
for i in range(10, 13, 1):
    results = {}
    index = i*6
    for key, result in [
        ("geneticU5K1P2Cr1", searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index, tests, use_single_point_crossover=False, mut_pow=1, hours=hours)),
        ("geneticU5K1P2Cr2", searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index+1, tests, use_single_point_crossover=False, mut_pow=1, hours=hours)),
        ("geneticU5K2P2Cr1", searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index+2, tests, use_single_point_crossover=False, mut_pow=2, hours=hours)),
        ("geneticU5K2P2Cr2", searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index+3, tests, use_single_point_crossover=False, mut_pow=2, hours=hours)),
        ("geneticU5K1P2Cr1Ordered", searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index+4, tests, use_single_point_crossover=False, mut_pow=1, hours=hours, 
            order_center=center)),
        ("geneticU5K1P2Cr2Ordered", searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index+5, tests, use_single_point_crossover=False, mut_pow=1, hours=hours,
            order_center=center)),
            ]:
        results[key] = searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, 
            N_donors, index, tests, use_single_point_crossover=False, mut_pow=1, hours=hours)

    data = {}
    for key in results:
        data[key] = results[key][2]
    print (results)
    print (data)
    plt.clf()
    dn_search_util.plotPerformance(data, [(2, 0, " validation"), (2, 1, " error")])
    plt.savefig("SearchSummery%d.png"%(i))
#searchRandomAndPlot("%s.kmc"%(abs_file_path), N_acceptors, N_donors, 15, 2, 50)
#for i in range(1, 15):
#    print ()