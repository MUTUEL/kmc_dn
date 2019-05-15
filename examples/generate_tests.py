'''
@author: Bram de Wilde (b.dewilde-1@student.utwente.nl), Indrek Klanberg (i.klanberg@student.utwente.nl)
'''
import os
import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import random
import json
import math
from validate_tests import getMeanAndStandardDeviation, compareVisualizeErrors


#%% Parameters

N = 30  # Number of acceptors
M = 4  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(1E6)
#res = 1  # Resolution of laplace grid

def getRandomKMC(voltage_range, N, M, xdim=1, ydim=1, zdim=0):
    # Define electrodes
    electrodes = np.zeros((8, 4))
    voltage_range = 300
    electrodes[0] = [0, ydim/4, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[1] = [0, 3*ydim/4, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[2] = [xdim, ydim/4, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[3] = [xdim, 3*ydim/4, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[4] = [xdim/4, 0, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[5] = [3*xdim/4, 0, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[6] = [xdim/4, ydim, 0, random.random()*2*voltage_range-voltage_range]
    electrodes[7] = [3*xdim/4, ydim, 0, 0]
    #%% Initialize simulation object
    kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, 0, electrodes = electrodes)
    return kmc

def generateTests(n, prefix, N, M):
    for i in range(n):
        kmc = getRandomKMC(300, N, M)
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "tests/%s/test%d.kmc"%(prefix, i)
        abs_file_path = os.path.join(script_dir, rel_path)
        #%% Profile code
        #kmc.python_simulation(hops=hops)
        kmc.go_simulation(hops=5*10**6, goSpecificFunction="wrapperSimulateRecord")
        print (kmc.time)
        print (kmc.electrode_occupation)
        print (kmc.current)
        kmc.saveSelf(abs_file_path)

def runOneTest(hops):
    print ("NEW TEST\n")
    kmc = getRandomKMC(300, N, M)
    currents_py = []
    print ("PYTHON\\n\n")
    for i in range(8):
        currents_py.append([])
    for i in range(10):
        kmc.python_simulation(hops=hops)
        #print (kmc.current)
        for j in range(8):
            currents_py[j].append(kmc.current[j])

    currents_go = []
    print ("GO\\n\n")
    for i in range(8):
        currents_go.append([])
    for i in range(10):
        kmc.go_simulation(hops=hops, goSpecificFunction="wrapperSimulateRecord")
        #print (kmc.current)
        for j in range(8):
            currents_go[j].append(kmc.current[j])
    py_better = 0

    mean_diff = 0
        
    for i in range(8):
        mean, std = getMeanAndStandardDeviation(currents_go[i]) 
        print ("GO: average: %.3g, std. deviation: %.3g\n"%(mean, std))
        
        mean2, std2 = getMeanAndStandardDeviation(currents_py[i])
        print ("PY: average: %.3g, std. deviation: %.3g\n"%(mean2, std2))
        if std2 < std:
            py_better+=1
        mean_diff += math.fabs(mean-mean2)/math.fabs(mean2)
    print ("Python was better %d out of 8 times, and average perc. difference was %.3g"%(py_better, mean_diff/8))

def testRandFunctions():
    kmc = getRandomKMC(300, N, M)
    data_rand = {}
    kmc.go_simulation(hops=100, goSpecificFunction="wrapperSimulate")
    with open('ExpRand.log') as f:
        data_rand['exp'] = json.load(f)
    with open('Ln1Rand.log') as f:
        data_rand['ln'] = json.load(f)
    data_rand['python'] = []
    for i in range(100000):
        data_rand['python'].append(np.random.exponential(scale=1/0.23))
    compareVisualizeErrors(data_rand, "rand_log.png")

def testOverlap():
    kmc = getRandomKMC(300, N, M)
    kmc.go_simulation(hops=1000000, goSpecificFunction="analyzeStateOverlap")

#for _ in range(10):
#    runOneTest(hops)

#testRandFunctions()
#testOverlap()

generateTests(200, "rnd4", 30, 3)