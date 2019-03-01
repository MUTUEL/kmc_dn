'''
@author: Bram de Wilde (b.dewilde-1@student.utwente.nl), Indrek Klanberg (i.klanberg@student.utwente.nl)
'''
import os
import math
import sys
sys.path.insert(0,'../')
import kmc_dopant_networks as kmc_dn
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import time
import kmc_dopant_networks_utils as kmc_utils

def getMeanAndStandardDeviation(arr):
    sum = 0
    arr.sort()
    print (arr[-25:])
    for ele in arr:
        sum+=ele
    average = sum / len(arr)
    sum = 0
    for ele in arr:
        sum+=math.fabs(ele-average)**2
    standardDeviation = math.sqrt(sum / len(arr))
    return average, standardDeviation

def getTests(prefix, amount):
    kmcs = []
    for j in range(amount):
        kmc = kmc_dn.kmc_dn(10, 3, 1, 1, 0)
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "tests/%s/test%d.kmc"%(prefix, j)
        abs_file_path = os.path.join(script_dir, rel_path)
        kmc.loadSelf(abs_file_path)
        kmcs.append(kmc)
    return kmcs

def evaluate(kmcs, printThreshold):
    count_negative_directions = 0
    sum_direction_sums = 0
    diffs = []
    j = 0
    extreme_errors = []
    for kmc in kmcs:
        direction_sum = 0.0
        for i in range(len(kmc.current)):
            diff = (math.fabs(kmc.current[i]-kmc.expected_current[i])/math.fabs(kmc.expected_current[i]))
            if diff > printThreshold:
                if len(extreme_errors) == 0 or extreme_errors[-1][0]!= j:
                    extreme_errors.append((j, "Expected current: %.3f, simulated current: %.3f, electrode is %d"%(kmc.expected_current[i], kmc.current[i], i)))
                    print("Extreme error")
            diffs.append(diff)
            direction = math.fabs(kmc.current[i]) - math.fabs(kmc.expected_current[i])
            direction_sum+=direction
        if direction_sum < 0:
            count_negative_directions+=1
        sum_direction_sums+=direction_sum
        j+=1
    average_error, error_SD = getMeanAndStandardDeviation(diffs)
    print("Average error: %.2f, Standard deviation: %.2f, direction sum: %.2f"%(average_error, error_SD, sum_direction_sums/len(kmcs)))
    return extreme_errors

def test(kmcs, func, title):
    print ("Starting testing with function: %s"%(title))
    start = time.time()

    for kmc in kmcs:
        func(kmc)
    end = time.time()
    print ("Took time: %.3f"%(end-start))
    extreme_errors = evaluate(kmcs, 0.7)
    return extreme_errors


def testKMC5000(kmc, record=False):
    kmc.go_simulation(hops=5000, record = record)
    
def testKMC50000(kmc, record=False):
    kmc.go_simulation(hops=50000, record = record)


def testKMC1E6(kmc, record=False):
    kmc.go_simulation(hops=1000000, record = record)

def testKMC100000(kmc, record=False):
    kmc.go_simulation(hops=100000, record=record)

def testProb1000(kmc, record=False):
    kmc.go_simulation(hops=1000, goSpecificFunction="wrapperSimulateProbability", record = record )

def testProb500(kmc, record=False):
    kmc.go_simulation(hops=500, goSpecificFunction="wrapperSimulateProbability", record = record)
def testProb5000(kmc, record=False):
    kmc.go_simulation(hops=5000, goSpecificFunction="wrapperSimulateProbability", record = record)

def testCombined10K(kmc, record=True):
    kmc.go_simulation(hops=10000, goSpecificFunction="wrapperSimulateCombined", record = True)

def visualize(kmc, func):
    func(kmc, True)
    kmc_utils.visualize_traffic(kmc)
    plt.show()


def compareVisualize(kmc, funcs, titles, fileName):
    prefixes = [11, 21, 22, 22, 32, 32, 33, 33, 33]
    plt.clf()
    fig = plt.figure(figsize=((prefixes[len(funcs)]/10*10), (prefixes[len(funcs)]%10)*10))
    for i in range(len(funcs)):
        funcs[i](kmc, True)
        sub_plot_number = prefixes[len(funcs)]*10+i+1
        fig = kmc_utils.visualize_traffic(kmc, sub_plot_number, titles[i], fig)
    plt.savefig(fileName)

def compareVisualizeSwipe(data, fileName):
    prefixes = [11, 21, 22, 22, 32, 32, 33, 33, 33]
    plt.clf()
    size = len(data)
    i = 1
    fig = plt.figure(figsize=((prefixes[size]/10*10), (prefixes[size]%10)*10))

    for key in data:
        sub_plot_number = prefixes[size]*10+i

        kmc_utils.plot_swipe(data[key], pos=sub_plot_number, figure=fig, title=key)
        i+=1
    plt.savefig(fileName)


def testSet(prefix, amount):
    tests = getTests(prefix, amount)
    print ("finished reading testSet %s"%(prefix))
    extreme_errors = {}
    for func, title in [
        #(testKMC5000, "KMC 5000 hops"), (testKMC50000, "KMC 50000 hops"), 
        #(testKMC1E6, "KMC 1E6 hops"), 
        #(testProb500, "Probability 500 hops"), (testProb1000, "Probability 1000 hops"), 
        #(testProb5000, "Probability 5000 hops"),
        (testCombined10K, "Combined 10K hops")]:
        extreme_errors[title] = (func, test(tests, func, title))
    for i in range(amount):
        funcs = [testKMC1E6]
        titles = ["BaseLine"]
        for title in extreme_errors:
            for j, error in extreme_errors[title][1]:
                if j == i:
                    funcs.append(extreme_errors[title][0])
                    titles.append("%s %s"%(title, error))
        if len(funcs) > 1:
            compareVisualize(tests[i], funcs, titles, "EV%s%d.png"%(prefix, i))

def measureSwipe(prefix, amount, inputVoltage, outPutCurrent, funcs):
    tests = getTests(prefix, amount)
    print("Finished reading %s\n"%(prefix))
    data = {}
    for _, title in funcs:
        data[title] = []
    for func, title in funcs:
        for kmc in tests:
            func(kmc)
            data[title].append((kmc.electrodes[inputVoltage][3], kmc.current[outPutCurrent]))
        print ("Finished simulating on %s\n"%(title))
    compareVisualizeSwipe(data, "Swipe%s.png"%(prefix))

testSet("set", 101)
testSet("xor", 100)
testSet("rnd", 200)
"""measureSwipe("xor", 100, 1, 2, [
        (testKMC5000, "KMC 5000 hops"), 
        #(testKMC50000, "KMC 50000 hops"), 
        #(testKMC1E6, "KMC 1E6 hops"), 
        #(testProb500, "Probability 500 hops"), (testProb1000, "Probability 1000 hops"), 
        #(testProb5000, "Probability 5000 hops"),
        (testCombined10K, "Combined 10K hops")]
        )"""