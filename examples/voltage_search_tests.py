import os
import sys
#sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import voltage_search
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import kmc_dopant_networks_utils as kmc_utils
import dn_search_util
import matplotlib.pyplot as plt
import pickle
from validate_tests import compareVisualizeErrorDistribution



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
    search = voltage_search.voltage_search(dn, 160, 10, tests)
    schedule = schedule_function(error_threshold_multiplier, hours)
    return search.simulatedAnnealingSearch(0.002*error_threshold_multiplier, schedule, "Voltage10DOP", animate=False)

def searchGeneticBasedOnTest(dn, tests, hours = 10, uniqueness = 5000, disparity=2, 
        mut_pow=1, order_center = None, gen_size = 50, index = 0):
    search = voltage_search.voltage_search(dn, 160, 10, tests)
    cross_over_function = search.singlePointCrossover
    return search.genetic_search(gen_size, 3600*hours, 2, uniqueness, "VoltageGenetic%d"%(index), 
        cross_over_function = cross_over_function, mut_pow=mut_pow, order_center=order_center)

for i in range(50, 60):
    dn = getRandomDn(30, 3)
    
    results = {}

    #results['annealing'] = searchAnnealing(dn, get_schedule1, [((0, 0), False), ((0, 100), True), ((100, 0), True), 
    #    ((100, 100), False)], 5, error_threshold_multiplier=20)

    results['genetic'] = searchGeneticBasedOnTest(dn, [((False, False), False), ((False, True), True), ((True, False), True), 
            ((True, True), False)], hours = 5, gen_size=200, index=i)

    data = {}
    for key in results:
        data[key] = results[key][2]
    print (results)
    print (data)
    plt.clf()
    dn_search_util.plotPerformance(data, [(2, 0, " validation"), (2, 1, " error")])
    plt.savefig("VoltageSearchSummery%d.png"%(i))