import os
import sys
#sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
from voltage_search import voltage_search
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import kmc_dopant_networks_utils as kmc_utils
import dn_search_util
import matplotlib.pyplot as plt
import pickle
from validate_tests import compareVisualizeErrorDistribution



def get8Electrodes(xdim, ydim):
    electrodes = np.zeros((8, 4))
    electrodes[0] = [0, 3*ydim/4, 0, 0]
    electrodes[1] = [xdim/4, 0, 0, 10]
    electrodes[2] = [xdim, ydim/4, 0, 10]
    electrodes[3] = [xdim, 3*ydim/4, 0, 0]
    electrodes[4] = [0, ydim/4, 0, 10]
    electrodes[5] = [3*xdim/4, 0, 0, 0]
    electrodes[6] = [xdim/4, ydim, 0, 10]
    electrodes[7] = [3*xdim/4, ydim, 0, 0]

    return electrodes

def getRandomDn(N_acceptors, N_donors):
    electrodes = get8Electrodes(1, 1)
    dn = kmc_dn.kmc_dn(N_acceptors, N_donors, 1, 1, 0, electrodes = electrodes)
    return dn


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

def searchAnnealing(dn, schedule_function, tests, hours = 10, error_threshold_multiplier = 1):
    search = voltage_search(dn, 160, 10, tests)
    schedule = schedule_function(error_threshold_multiplier, hours)
    return search.simulatedAnnealingSearch(0.002*error_threshold_multiplier, schedule, "Voltage10DOP", animate=False)

def searchGeneticBasedOnTest(dn, tests, hours = 10, uniqueness = 1000, disparity=2, 
        mut_pow=1, order_center = None, gen_size = 50, index = 0):
    search = voltage_search(dn, 150, 10, tests, corr_pow=2, parallelism=25)
    cross_over_function = search.singlePointCrossover
    results = search.genetic_search(gen_size, 3600*hours, 2, uniqueness, cross_over_function = cross_over_function, mut_pow=mut_pow, order_center=order_center, max_generations=20)
    search.best_dn.genetic_search_results = search.validations
    search.saveResults(True, False, "resultDump", index)
    return results

def searchSPSA(dn, tests, hours = 1, index = 0):
    search = voltage_search(dn, 300, 10, tests)
    return search.SPSA_search(hours*3600, a=500, c=10, A=100, alfa=0.5, gamma=0.2, file_prefix="VoltageSPSA%d"%(index))

def testVC(dn, dim, points, starting_index, prefix=""):
    for case in range(1, (2**dim)-1):
        tests = []
        results = {}
        for i in range(dim):
            tests.append((points[i], case&(2**i)))
        print (tests)
        results['genetic'] = searchGeneticBasedOnTest(dn, tests, hours = 1, gen_size=100, 
        index=case+starting_index)
        data = {}
        for key in results:
            data[key] = results[key][2]
        plt.clf()
        dn_search_util.plotPerformance(data, [(2, 0, " validation"), (2, 1, " error")])
        plt.savefig("%sVCdim%dCase%d.png"%(prefix, dim, case))

def reTestVC(dn, dim, points, cases, starting_index, prefix=""):
    for case in cases:
        tests = []
        results = {}
        for i in range(dim):
            tests.append((points[i], case&(2**i)))
        print (tests)
        results['genetic'] = searchGeneticBasedOnTest(dn, tests, hours = 1, gen_size=100, 
        index=case+starting_index)
        data = {}
        for key in results:
            data[key] = results[key][2]
        plt.clf()
        dn_search_util.plotPerformance(data, [(2, 0, " validation"), (2, 1, " error")])
        plt.savefig("%sVCdim%dCase%d.png"%(prefix, dim, case))

#rel_path = "../GeneticResultDumpVoltageGenetic1.kmc"
# script_dir = os.path.dirname(__file__)
# abs_file_path = os.path.join(script_dir, rel_path)
# dn.loadSelf(abs_file_path)
#points = [(-150, -150), (-150, 150), (150, -150), (150, 150), (-50, 0), (50, 0)]
points = [(0, 0), (0, 75), (75, 0), (75, 75)]#, (-50, 0), (50, 0)]

# reV4 = [2, 10]
# reV5 = [8, 23]
# reV6 = [2, 8, 14, 19, 26, 27, 34, 38, 49]

#testVC(dn, 4, points, 8000, prefix="30DOPTRY2CP2")

dops = [5, 8, 10, 20, 30, 45, 60]
startIndex = 0
times = 100
timeProfile = {}
for dop in dops:
    timeProfile[dop] = []
    for index in range(times):
        start = time.time()
        dn = getRandomDn(dop, round(dop/10))
        reTestVC(dn, 4, points, [6], startIndex+index, prefix="%dDOPTRY%dCP2"%(dop, index))
        end = time.time()
        diff = end - start
        timeProfile[dop].append(diff)
    startIndex+=times
print (timeProfile)

#testVC(dn, 5, points, 6016, prefix="10DOPTRY5CP2")
#testVC(dn, 6, points, 1048, prefix="10DOP")
# reTestVC(dn, 4, points, reV4, 0)
# reTestVC(dn, 5, points, reV5, 16)
# reTestVC(dn, 6, points, reV6, 48)
