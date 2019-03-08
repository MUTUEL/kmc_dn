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
            volt = math.fabs(volt)/volt*volt*volt*400
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

def searchBasedOnTest(fileName):
    with open(fileName, 'rb') as f:
        tests = pickle.load(f)
        dn = getRandomDn(20, 2)
        search = dn_search.dn_search(dn, tests, 1, 1, 0.04, 0.04)
        search.simulatedAnnealingSearch(0.08, 2000)

    
rel_path = "search_tests/test_set3"
abs_file_path = os.path.join(os.path.dirname(__file__), rel_path)
#genAndSaveTest(abs_file_path, 20, 2, 10)
searchBasedOnTest("%s.kmc"%(abs_file_path))