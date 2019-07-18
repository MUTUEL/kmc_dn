import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import kmc_dopant_networks as kmc_dn
import os
import sys
import math
from utils import openKmc

from matplotlib.ticker import FormatStrFormatter


def drawGraph(dn, gs, fig, index):
    ax = plt.subplot(gs)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
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
        
    y_data = []
    y_data_expected = []
    acs = 0
    for entry in tmp_swipe_results:
        curr = entry[1][-1]
        for c in entry[1]:
            acs+=abs(c)
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

    Eac = acs / (len(tmp_swipe_results) * len (tmp_swipe_results[0][1]))
    x_data = [i for i in range(len(y_data))]
    ax.plot(x_data, y_data)
    ax.plot(x_data, y_data_expected, color='k')
    ax.set_xticklabels([])
    y = y_data_min - 0.2*(y_data_max - y_data_min)
    sepa = true_min - false_max
    ax.text(40, y, "(%d) Separation: %.3g"%(index, sepa), fontdict={"fontsize": 16,})
    return sepa, Eac



def compileGraphs(indexes, fileOut, key, prefix="swipeResults/xor"):
    y_size = math.ceil(len(indexes)/3.0)
    fig = plt.figure(figsize=(16, y_size*3))
    gs = gridspec.GridSpec(y_size, 3)   
    plt.subplots_adjust(wspace=0.3)
    
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    i = 0
    separated = 0
    sEc = 0
    for index in indexes:
        rel_path = "%s%d.kmc"%(prefix, index)
    
        abs_file_path = os.path.join(script_dir, rel_path)
        dn = openKmc(abs_file_path)
        separation, Ec = drawGraph(dn, gs[i], fig, i+1)
        sEc+=Ec
        if  separation > 0:
            separated+=1

        i+=1
    Ec = sEc / len(indexes)
    EcA = Ec / 10**(-6)
    print ("%s| Separation found: %d/%d, average current: %.3g, average supposed physical current:%.1fnA"%(key, separated, len(indexes), Ec, EcA))
    plt.savefig(fileOut)


def main():
    index5 = [i for i in range(8, 20)]
    index5.extend([6, 126])
    indexes = {
        "30Dvc4": [i for i in range(1, 15)],
        # "30Dvc5":[i for i in range(17, 47)],
        # "30Dvc6":[i for i in range(49, 112)],
        # "20Dvc4": [i for i in range(1001, 1015)],
        # "20Dvc5":[i for i in range(1017, 1047)],
        # "10D.1vc4": [i for i in range(2001, 2015)],
        # "10D.1vc5": [i for i in range(2017, 2047)],
        # "10D.2vc4": [i for i in range(3001, 3015)],
        # "10D.2vc5": [i for i in range(3017, 3047)],
        # "10D.3vc4": [i for i in range(4001, 4015)],
        # "10D.3vc5": [i for i in range(4017, 4047)],
        "10D4vc4": [i for i in range(5001, 5015)],
        # "10D.4vc5": [i for i in range(5017, 5047)],
    }
    # indexes = {
    #     "D20vc4":[i for i in range(1001, 1015)],
    #     "D20vc5":[i for i in range(1017, 1047)],
    # }
    for key in indexes:
        compileGraphs(indexes[key], "Summery%s.png"%(key), key)

if __name__== "__main__":
  main()