import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import kmc_dopant_networks as kmc_dn
import os
import sys
import math

def openKmc(abs_file_path):
    '''
    This returns a kmc_dn object that is in the given absolute file path.
    :param abs_file_path:
    :return:
    '''
    kmc = kmc_dn.kmc_dn(10, 3, 1, 1, 0)
    kmc.loadSelf(abs_file_path)
    return kmc

def drawGraph(dn, gs, fig, index):
    ax = plt.subplot(gs)
    y_data_max = -1
    y_data_min = 1
    false_max = -1
    true_min = 1
    tmp_swipe_results = []
    
    allowed_indexes = [i for i in range(0, 40)]
    allowed_indexes.extend(range(80, 120))
    allowed_indexes.extend(range(160, 200))
    allowed_indexes.extend(range(240, 280))
    allowed_indexes.extend(range(320, 360))
    allowed_indexes.extend(range(400, 440))

    for i in allowed_indexes:
        if i < len(dn.swipe_results):
            tmp_swipe_results.append(dn.swipe_results[i])

    print (len(tmp_swipe_results))
        
    y_data = []
    y_data_expected = []
    for entry in tmp_swipe_results:
        curr = entry[1][7]
        expected = entry[4]
        y_data.append(curr)
        if expected == 1 and true_min > curr:
            true_min = curr
        if expected == 0 and false_max < curr:
            false_max = curr

        if curr < y_data_min:
            y_data_min = curr
        if curr > y_data_max:
            y_data_max = curr
        
    for entry in tmp_swipe_results:
        expected = entry[4]
        y_data_expected.append(y_data_min + (y_data_max-y_data_min)*expected)

    x_data = [i for i in range(len(y_data))]
    ax.plot(x_data, y_data)
    ax.plot(x_data, y_data_expected, color='k')
    ax.set_xticklabels([])
    y = y_data_min - 0.2*(y_data_max - y_data_min)
    sepa = true_min - false_max
    ax.text(40, y, "(%d) Separation: %.3g"%(index, sepa))
    return sepa



def compileGraphs(indexes, fileOut):
    y_size = math.ceil(len(indexes)/3.0)
    fig = plt.figure(figsize=(12, y_size*3))
    gs = gridspec.GridSpec(y_size, 3)
    
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    i = 0
    separated = 0
    for index in indexes:
        rel_path = "swipeResults/xor%d.kmc"%(index)
    
        abs_file_path = os.path.join(script_dir, rel_path)
        dn = openKmc(abs_file_path)
        if drawGraph(dn, gs[i], fig, i+1) > 0:
            separated+=1

        i+=1
    print ("%d/%d"%(separated, len(indexes)))
    plt.savefig(fileOut)


def main():
    index5 = [i for i in range(8, 20)]
    index5.extend([6, 126])
    indexes = {
        #"vc4":[i for i in range(1, 15)],
        "vc5":[i for i in range(17, 47)],
        "vc6":[i for i in range(49, 112)],
    }
    # indexes = {
    #     "D20vc4":[i for i in range(1001, 1015)],
    #     "D20vc5":[i for i in range(1017, 1047)],
    # }
    for key in indexes:
        compileGraphs(indexes[key], "Summery%s.png"%(key))

if __name__== "__main__":
  main()