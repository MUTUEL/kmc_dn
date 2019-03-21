import os
import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import dn_search
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import kmc_dopant_networks_utils as kmc_utils
import matplotlib.pyplot as plt
import pickle


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
            (0.2*multiplier, 500*time_multiplier, 0),
            (0.1*multiplier, 950*time_multiplier, 0),
            (0.1*multiplier, 955*time_multiplier, 1),
            (0.15*multiplier, 1300*time_multiplier, 1),
            (0.05*multiplier, 1750*time_multiplier, 1),
            (0.02*multiplier, 2000*time_multiplier, 1),
            (0.01*multiplier, 2500*time_multiplier, 1),
            (0, 2700*time_multiplier, 2),
        ]

def get_schedule2(multiplier, time_multiplier):
    return [
        (0.2*multiplier, 1500*time_multiplier, 0),
        (0, 1505*time_multiplier, 1),
        (0.1*multiplier, 2500*time_multiplier, 1),
        (0, 2700*time_multiplier, 1)
    ]

def searchBasedOnTest(fileName, N_acceptors, N_donors, test_index, schedule_function):
    with open(fileName, 'rb') as f:
        tests = pickle.load(f)
        dn = getRandomDn(N_acceptors, N_donors)
        search = dn_search.dn_search(dn, tests, 1, 1, 0.04, 0.04)
        multiplier = 2
        time_multiplier = 2
        schedule = schedule_function(multiplier, time_multiplier)
        return search.simulatedAnnealingSearch(0.4, schedule, "10DOP%d"%(test_index))
    return None, None

def searchGeneticBasedOnTest(fileName, N_acceptors, N_donors, test_index):
    with open(fileName, 'rb') as f:
        tests = pickle.load(f)
        dn = getRandomDn(N_acceptors, N_donors)
        search = dn_search.dn_search(dn, tests, 1, 1, 0.04, 0.04)

        return search.genetic_search(100, 100, [], 2, 5000, "10DOP%d"%(test_index))
    return None, None


N_acceptors = 10
N_donors = 3
rel_path = "search_tests/test_set_4"
abs_file_path = os.path.join(os.path.dirname(__file__), rel_path)
#genXorTest(abs_file_path, "xor_example.png")

#genAndSaveTest(abs_file_path, N_acceptors, N_donors, 2)
"""results = {'schedule_1':[], 'schedule_2':[]}
for i in range(5):
    result, strategy = searchBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, N_donors, 12+i, get_schedule1)
    results['schedule_1'].append((result, strategy))

for i in range(5):
    result, strategy = searchBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, N_donors, 18+i, get_schedule2)
    results['schedule_2'].append((result, strategy))
print (results)"""
searchGeneticBasedOnTest("%s.kmc"%(abs_file_path), N_acceptors, N_donors, 101)