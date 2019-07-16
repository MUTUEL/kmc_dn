'''
@author: Indrek Klanberg (i.klanberg@student.utwente.nl)
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
import seaborn as sns

def getMeanAndStandardDeviation(arr):
    sum = 0
    arr.sort()
    #print (arr[-25:])
    for ele in arr:
        sum+=ele
    average = sum / len(arr)
    sum = 0
    for ele in arr:
        sum+=math.fabs(ele-average)**2
    standardDeviation = math.sqrt(sum / len(arr))
    return average, standardDeviation

def getTests(prefix, amount, start = 0):
    kmcs = []
    for j in range(start, amount+start):
        kmc = kmc_dn.kmc_dn(10, 3, 1, 1, 0)
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = "tests/%s/test%d.kmc"%(prefix, j)
        abs_file_path = os.path.join(script_dir, rel_path)
        kmc.loadSelf(abs_file_path)
        kmcs.append(kmc)
    return kmcs

def evaluate_single(kmcs, printThreshold):
    count_negative_directions = 0
    sum_direction_sums = 0
    diffs = []
    j = 0
    extreme_errors = []
    for kmc in kmcs:
        direction_sum = 0.0
        for i in range(len(kmc.current)):
            diff = math.fabs(kmc.current[i]-kmc.expected_current[i])
            if diff > printThreshold:
                if len(extreme_errors) == 0 or extreme_errors[-1][0]!= j:
                    extreme_errors.append((j, "Expected current: %.3f, simulated current: %.3f, electrode is %d"%(kmc.expected_current[i], kmc.current[i], i)))
            
            direction_multi = 1 if (kmc.current[i] < 0) == (kmc.expected_current[i] < 0) else -1
            direction = math.fabs(kmc.current[i]) - direction_multi*math.fabs(kmc.expected_current[i])
            direction_sum+=direction
            #if math.fabs(kmc.expected_current[i]) >= 0.1:
            diffs.append(diff)
        if direction_sum < 0:
            count_negative_directions+=1
        sum_direction_sums+=direction_sum
        j+=1
    average_error, error_SD = getMeanAndStandardDeviation(diffs)
    print("Average error: %.2g, Standard deviation: %.2g, direction sum: %.2g"%(average_error, error_SD, sum_direction_sums/len(kmcs)))
    return extreme_errors, diffs

def test_single(kmcs, func, title):
    print ("Starting testing with function: %s"%(title))
    start = time.time()

    for kmc in kmcs:
        func(kmc)
    end = time.time()
    print ("Took time: %.3f"%(end-start))
    extreme_errors, diffs = evaluate_single(kmcs, 0.7)
    return extreme_errors, diffs

def calc_D(meanp, meanv, std_p, std_v):
    var_p = std_p**2
    var_v = std_v**2
    if var_p == 0 or var_v == 0:
        return math.log(10000)
    part1 = 0.25 * math.log(0.25*(var_p/var_v + var_v/var_p + 2))
    part2 = 0.25 * (((meanp - meanv)**2) / (var_p + var_v))
    return part1 + part2


def evaluate_BC(kmcs, printThreshold):
    diffs = []
    j = 0
    extreme_errors = []
    for kmc in kmcs:
        for i in range(len(kmc.current)):
            diff = calc_D(kmc.mean_currents[i], kmc.new_means_c[i], kmc.stddev_currents[i], kmc.new_std_deviation_c[i])
            if diff > printThreshold:
                if len(extreme_errors) == 0 or extreme_errors[-1][0]!= j:
                    extreme_errors.append((j, "Expected current: %.3f, simulated current: %.3f, electrode is %d"%(kmc.expected_current[i], kmc.current[i], i)))
            
            diffs.append(diff)
        j+=1
    average_error, error_SD = getMeanAndStandardDeviation(diffs)
    print("Average error: %.2g, Standard deviation: %.2g"%(average_error, error_SD))
    return extreme_errors, diffs



def calc_mean_and_std(kmc, func):
    currents = []
    for i in range(5):
        func(kmc)
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
    kmc.new_means_c = means_c
    kmc.new_std_deviation_c = std_deviation_c

def test_BC(kmcs, func, title):
    print ("Starting testing with function: %s"%(title))
    start = time.time()

    for kmc in kmcs:
        calc_mean_and_std(kmc, func)
    end = time.time()
    print ("Took time: %.3f"%(end-start))
    extreme_errors, diffs = evaluate_BC(kmcs, 0.9)
    return extreme_errors, diffs

def testPython5K(kmc, record=False):
    kmc.python_simulation(hops=5000)

def testCombined10K(kmc, record=True):
    kmc.go_simulation(hops=10000, goSpecificFunction="wrapperSimulateCombined", record = True)
def getPythonFunction(hops, prehops = 0):
    def new_func(kmc, record=True):
        kmc.python_simulation(hops=hops, prehops=prehops, record = record)
    return new_func, "Python KMC with %d hops, %d prehops"%(hops, prehops)

def getKMCFunctionNoRecord(hops, prehops = 0):
    def new_func(kmc, record=True):
        kmc.go_simulation(hops=hops, prehops=prehops, record = record, goSpecificFunction="wrapperSimulate")
    return new_func, "Go KMC No record with %d hops, %d prehops"%(hops, prehops)

def getKMCFunctionRecordPlus(hops, prehops = 0, index=0):
    def new_func(kmc, record=False):
        kmc.go_simulation(hops=hops, prehops=prehops, record = record, goSpecificFunction="wrapperSimulateRecordPlus")
    return new_func, "KMC record+ with %d hops, %d prehops:%d"%(hops, prehops, index)

def getKMCFunctionRecordAndAverage(hops, prehops = 0, index=0):
    def new_func(kmc, record=False):
        kmc.go_simulation(hops=hops, prehops=prehops, record = record, goSpecificFunction="wrapperSimulateRecordWithAverage")
    return new_func, "KMC record+average %d hops, %d prehops:%d"%(hops, prehops, index)


def getKMCFunction(hops, prehops = 0):
    def new_func(kmc, record=False):
        kmc.go_simulation(hops=hops, prehops=prehops, record = record, goSpecificFunction="wrapperSimulateRecord")
    return new_func, "KMC with %d hops, %d prehops"%(hops, prehops)
def getProbFunction(hops):
    def new_func(kmc, record=True):
        kmc.go_simulation(hops=hops, goSpecificFunction="wrapperSimulateProbability", record = record)
    return new_func, "Probability occupation with %d hops"%(hops)
def getPruneFunction(hops, prune_threshold):
    def new_func(kmc, record=True):
        kmc.go_simulation(hops=hops, goSpecificFunction="wrapperSimulatePruned", record = True, prune_threshold=prune_threshold)
    return new_func, "Python pruned KMC %d hops, %.2g threshold"%(hops, prune_threshold)

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
    size = len(data)-1
    i = 1
    fig = plt.figure(figsize=((prefixes[size]/10*10), (prefixes[size]%10)*10))

    for key in data:
        sub_plot_number = prefixes[size]*10+i
        kmc_utils.plot_swipe(data[key], pos=sub_plot_number, figure=fig, title=key)
        i+=1    
    plt.savefig(fileName)

def compareVisualizeErrors(diffs, fileName, x_limit = 0):
    prefixes = [11, 21, 22, 22, 32, 32, 33, 33, 33, 43, 43, 43, 53, 53, 53]
    plt.clf()
    prefix = prefixes[len(diffs)-1]
    index = 1
    fig = plt.figure(figsize=(math.floor(prefix/10)*10, (prefix%10)*10))

    for title in diffs:
        avg, dev = getMeanAndStandardDeviation(diffs[title])
        if x_limit == 0:
            x_dim = avg + 3*dev
        else:
            x_dim = x_limit
        sub_plot_number = plt.subplot(prefix/10, prefix%10,index)
        diffs[title].sort()
        ordered_tuple = []
        i = len(diffs[title])
        for ele in diffs[title]:
            ordered_tuple.append((ele, i))
            i-=1
        ax = kmc_utils.plot_swipe(ordered_tuple, sub_plot_number, fig, title, xlim=x_dim)
        avg, dev = getMeanAndStandardDeviation(diffs[title])
        ax.text(0.5, -0.1, "Average: %.3g, Std. deviation: %.3g"%(avg, dev), horizontalalignment='center', transform=ax.transAxes)

        index+=1
    plt.savefig(fileName)

def compareVisualizeErrorDistribution(diffs, fileName, x_limit=0):
    prefixes = [11, 21, 22, 22, 32, 32, 33, 33, 33, 43, 43, 43, 53, 53, 53]
    plt.clf()
    prefix = prefixes[len(diffs)-1]
    index = 1
    fig = plt.figure(figsize=((prefix%10)*10, math.floor(prefix/10)*10))

    for title in diffs:
        sub_plot_number = plt.subplot(prefix/10, prefix%10,index)
        avg, dev = getMeanAndStandardDeviation(diffs[title])
        if x_limit == 0:
            x_dim = avg + 3*dev
        else:
            x_dim = x_limit
        diffs[title].sort()
        diff_list=diffs[title]
        for i in range(len(diff_list)):
            if diff_list[i] > x_dim:
                break
        if i < len(diff_list) - 1:
            diff_list = diff_list[:i]

        ax = fig.add_subplot(sub_plot_number)

        ax.set_xlim(right=x_dim)
        ax.set_title(title)
        
        ax.text(0.5, -0.1, "Average: %.3g, Std. deviation: %.3g"%(avg, dev), horizontalalignment='center', transform=ax.transAxes)
        bins = 50
        sns.distplot(diff_list, kde=True, ax=ax, bins=bins,
             color = 'darkblue',
             norm_hist=True,
             kde_kws={'linewidth': 4})
        index+=1
    plt.savefig(fileName)
def findWierdnessInTestSet(prefix, amount):
    tests = getTests(prefix, amount)
    print ("finished reading testSet %s"%(prefix))
    i = 0
    all_results = []
    keys_printed = False
    for kmc in tests:
        results = {}
        for func, title in [
            getKMCFunction(1000000),
            getKMCFunctionNoRecord(1000000),
            getPythonFunction(1000000),
            ]:
            func(kmc)
            results[title] = ["%.3g"%(math.fabs(kmc.current[j]-kmc.expected_current[j])) for j in range(len(kmc.current))]
            results[title][-1] = results[title][-1]+"\n"
        
        for i in range(len(kmc.current)):
            result = [kmc.expected_current[i]]
            for key in results:
                result.append(results[key][i])
            all_results.append(result)
            if not keys_printed:
                for key in results:
                    print (key)
                    keys_printed = True
    all_results = sorted(all_results, key=lambda x:x[0])
    for res in all_results:
        print ("%s\n"%(str(res)))

def testSet(prefix, amount, start = 0):
    tests = getTests(prefix, amount, start)
    print ("finished reading testSet %s"%(prefix))
    errors = {}
    diffs = {}
    extreme_errors = {}
    # diffs['values'] = []
    # for kmc in tests:
    #     for curr in kmc.expected_current:
    #         diffs['values'].append(math.fabs(curr))

    for func, title in [
        # getKMCFunctionRecordAndAverage(100000),
        # getKMCFunctionRecordAndAverage(500000),
        # getKMCFunctionRecordAndAverage(1000000),
        # getKMCFunctionRecordAndAverage(2000000),
        # getKMCFunctionRecordAndAverage(3000000),
        # getKMCFunctionRecordAndAverage(4000000),
        # getKMCFunctionRecordAndAverage(5000000),
        # getKMCFunctionRecordPlus(100000),
        # getKMCFunctionRecordPlus(500000),
        getKMCFunctionRecordPlus(1000000),
        getKMCFunctionRecordPlus(5000000),
        # getPruneFunction(1000000, 0.00001),
        # getPruneFunction(1000000, 0.0000001),
        # getPruneFunction(1000000, 0.000000001),
        getPythonFunction(1000000),
        getKMCFunctionNoRecord(1000000),

        # getKMCFunctionNoRecord(1000000),
        # getPythonFunction(1000000),
        # getProbFunction(500),
        # getProbFunction(1000),
        # getProbFunction(2000),
        # getProbFunction(5000),
        # getProbFunction(25000),
        # getProbFunction(100000),
        # getProbFunction(1000000),
        #(testCombined10K, "Combined 10K hops"),
        
        #(getPruneFunction(25000, 0.00000001), "Python pruned KMC 25K hops, 0.000001% threshold"),
        #(testKMC1E6, "KMC 1E6 hops"), 
        ]:
        errors[title] = (func, test_BC(tests, func, title))
        diffs[title] = errors[title][1][1]
        extreme_errors[title] = (func, errors[title][1][0])
    compareVisualizeErrors(diffs, "Errors%s.png"%(prefix), 1)
    compareVisualizeErrorDistribution(diffs, "Kernel%s.png"%(prefix), 1)
    # for i in range(amount):
    #     funcs = [getKMCFunction(1000000)[0]]
    #     titles = ["BaseLine"]
    #     for title in extreme_errors:
    #         for j, error in extreme_errors[title][1]:
    #             if j == i:
    #                 funcs.append(extreme_errors[title][0])
    #                 titles.append("%s %s"%(title, error))
    #     if len(funcs) > 1:
    #        compareVisualize(tests[i], funcs, titles, "EV%s%d.png"%(prefix, i))

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



def main():
    #testSet("set", 101)
    #testSet("xor", 100)
    #testSet("rnd", 200)
    #testSet("rnd2", 200)
    testSet("rnd_min_max5M", 100)
    testSet("XOR_wide5M", 100)
    #testSet("rnd_min_max", 100)
    #testSet("XOR_wide", 100)
    #findWierdnessInTestSet("rnd2", 5)
    """measureSwipe("xor", 100, 1, 2, [
            getKMCFunction(5000),
            getKMCFunction(1000000),            
            getProbFunction(2500),
            #(testCombined10K, "Combined 10K hops"),
            #(getPruneFunction(1000000, 0.01), "Python pruned KMC 1M hops,1% threshold"),
            #(getPruneFunction(1000000, 0.003), "Python pruned KMC 1M hops, 0.3% threshold"),
            ])"""
  
if __name__== "__main__":
  main()


