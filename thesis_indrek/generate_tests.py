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
M = 3  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(1E6)
#res = 1  # Resolution of laplace grid

def getRandomKMC(voltage_range, N, M, xdim=1, ydim=1, zdim=0):
    # Define electrodes
    electrodes = np.zeros((8, 4))
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

def testKmc(kmc, use_python=False):
    currents = []
    for i in range(5):
        if use_python:
            kmc.python_simulation(hops=hops)
        else:
            kmc.go_simulation(hops=hops, goSpecificFunction="wrapperSimulateRecordPlus")
        currents.append(kmc.current.copy())
    means_c = []
    std_deviation_c = []
    for i in range(len(kmc.current)):
        arr = []
        for j in range(0, len(currents)):
            arr.append(currents[j][i])
        mean, std_dev = getMeanAndStandardDeviation(arr)
        means_c.append(mean)
        std_deviation_c.append(std_dev)
    kmc.mean_currents = means_c
    kmc.stddev_currents = std_deviation_c
    print (currents)
    print (means_c)
    print (std_deviation_c)

def generateTests(n, prefix, N, M):
    for i in range(n):
        print (i)
        kmc = getRandomKMC(300, N, M)
        rel_path = "tests/%s/test%d.kmc"%(prefix, i)
        testKmc(kmc)
        kmc.saveSelf(rel_path, True)

def generateTestsFromTests(n, prefix, from_prefix, use_python=False):
    for i in range(n):
        print (i)
        kmc = getRandomKMC(300, 30, 3)
        rel_path = "examples/tests/%s/test%d.kmc"%(from_prefix, i)
        kmc.loadSelf(rel_path, True)
        rel_path = "examples/tests/%s/test%d.kmc"%(prefix, i)
        testKmc(kmc, use_python=use_python)
        kmc.saveSelf(rel_path, True)


def generateXORTests(indexRange, folder):
    points = [(0, 0), (0, 75), (75, 0), (75, 75)]
    testCase = 1
    for i in indexRange:
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "resultDump%d.kmc"%(i)
        abs_file_path = os.path.join(script_dir, rel_path)
        kmc = getRandomKMC(300, N, M)
        kmc.loadSelf(abs_file_path)

        for p in points:
            kmc.electrodes[0][3] = p[0]
            kmc.electrodes[1][3] = p[1]
            testKmc(kmc, use_python=True)
            rel_path = "examples/tests/%s/test%d.kmc"%(folder, testCase)
            abs_file_path = os.path.join(script_dir, rel_path)
            kmc.saveSelf(abs_file_path)
            testCase+=1

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

#generateXORTests([i for i in range(500, 525)], "XOR_wide")
#generateTests(100, rnd_min_max", 30, 3)
generateTestsFromTests(100, "XOR_wide", "XOR_wide5M", True)
