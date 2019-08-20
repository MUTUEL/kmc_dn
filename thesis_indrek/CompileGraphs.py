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


# This file is used to generate Graphs from multiple .kmc files
# NB! These .kmc files need to have the swipe_results attribute inside them, which 
# is calculated in either AnimateSwipe or SaveResults script. Note that AnimateSwipe saves
# .kmc to another tests/XOR/ directory.

def drawGraph(dn, gs, fig, index, get_all_electrode_EC=True):
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

    if not hasattr(dn, "swipe_results"):
        print (index)
        return 0, 0

    for i in allowed_indexes:
        if i < len(dn.swipe_results):
            tmp_swipe_results.append(dn.swipe_results[i])
        
    y_data = []
    y_data_expected = []
    acs = 0
    for entry in tmp_swipe_results:
        curr = entry[1][-1]
        if get_all_electrode_EC:
            for c in entry[1]:
                acs+=abs(c)
        else:
            acs+=abs(curr)
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

    if get_all_electrode_EC:
        Eac = acs / (len(tmp_swipe_results) * len (tmp_swipe_results[0][1]))
    else:
        Eac = acs / len(tmp_swipe_results)
    x_data = [i for i in range(len(y_data))]
    ax.plot(x_data, y_data)
    ax.plot(x_data, y_data_expected, color='k')
    ax.set_xticklabels([])
    y = y_data_min - 0.2*(y_data_max - y_data_min)
    sepa = true_min - false_max
    ax.text(40, y, "(%d) Separation: %.3g"%(index, sepa), fontdict={"fontsize": 16,})
    return sepa, Eac



def compileGraphs(indexes, fileOut, key, prefix="swipeResults/xor", columns=3, get_all_electrode_EC=True):
    y_size = math.ceil(len(indexes)/columns)
    fig = plt.figure(figsize=(columns*5, y_size*3))
    gs = gridspec.GridSpec(y_size, columns)   
    plt.subplots_adjust(left=0.1, right=0.97, wspace=0.3, top=0.97, bottom=0.05)
    
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    i = 0
    separated = 0
    sEc = 0
    for index in indexes:
        rel_path = "%s%d.kmc"%(prefix, index)
    
        abs_file_path = os.path.join(script_dir, rel_path)
        dn = openKmc(abs_file_path)
        separation, Ec = drawGraph(dn, gs[i], fig, i+1, get_all_electrode_EC)
        sEc+=Ec
        if  separation > 0:
            separated+=1

        i+=1
    Ec = sEc / len(indexes)
    EcA = Ec / 10**(-6)
    print ("%s| Separation found: %d/%d, average current: %.3g, average supposed physical current:%.1fnA"%(key, separated, len(indexes), Ec, EcA))
    plt.savefig(fileOut)


def main():
    # index5 = [i for i in range(8, 20)]
    # index5.extend([6, 126])
    # indexes = {
    #     "30Dvc4": [i for i in range(1, 15)],
    #     # "30Dvc5":[i for i in range(17, 47)],
    #     # "30Dvc6":[i for i in range(49, 112)],
    #     # "20Dvc4": [i for i in range(1001, 1015)],
    #     # "20Dvc5":[i for i in range(1017, 1047)],
    #     # "10D.1vc4": [i for i in range(2001, 2015)],
    #     # "10D.1vc5": [i for i in range(2017, 2047)],
    #     # "10D.2vc4": [i for i in range(3001, 3015)],
    #     # "10D.2vc5": [i for i in range(3017, 3047)],
    #     # "10D.3vc4": [i for i in range(4001, 4015)],
    #     # "10D.3vc5": [i for i in range(4017, 4047)],
    #     "10D4vc4": [i for i in range(5001, 5015)],
    #     # "10D.4vc5": [i for i in range(5017, 5047)],
    # }
    # # indexes = {
    # #     "D20vc4":[i for i in range(1001, 1015)],
    # #     "D20vc5":[i for i in range(1017, 1047)],
    # # }
    # indexes = {
    #     "5DOP": index5,
    #     "10DOP": [i for i in range(46, 60)],
    #     "20DOP": [i for i in range(60, 74)],
    #     "30DOP": [i for i in range(74, 88)],
    #     "45DOP": [i for i in range(88, 102)],
    #     "60DOP": [i for i in range(102, 116)],
    # }
    # indexes = {
    #     #"5DOP": [i for i in range(6, 106)],
    #     #"10DOP": [i for i in range(106, 206)],
    #     #"20DOP": [i for i in range(206, 306)],
    #     #"30DOP": [i for i in range(306, 406)],
    #     #"45DOP": [i for i in range(406, 506)],
    #     "60DOP": [i for i in range(506, 606)],
    # }
    # indexes = {
    #     "30D.1vc4": [i for i in range(1, 15)],
    #     "30D.1vc5":[i for i in range(16, 46)],
    #     "30D.2vc4": [i for i in range(101, 115)],
    #     "30D.2vc5":[i for i in range(116, 146)],
    #     "30D.3vc4": [i for i in range(201, 215)],
    #     "30D.3vc5":[i for i in range(216, 246)],
    #     "30D.4vc4": [i for i in range(301, 315)],
    #     "30D.4vc5":[i for i in range(316, 346)],
    #     "20D.1vc4": [i for i in range(401, 415)],
    #     "20D.1vc5":[i for i in range(416, 446)],
    #     "20D.2vc4": [i for i in range(501, 515)],
    #     "20D.2vc5":[i for i in range(516, 546)],
    #     "20D.3vc4": [i for i in range(601, 615)],
    #     "20D.3vc5":[i for i in range(616, 646)],
    #     "20D.4vc4": [i for i in range(701, 715)],
    #     "20D.4vc5":[i for i in range(716, 746)],
    #     "10D.1vc4": [i for i in range(801, 815)],
    #     "10D.1vc5": [i for i in range(816, 846)],
    #     "10D.2vc4": [i for i in range(901, 915)],
    #     "10D.2vc5": [i for i in range(916, 946)],
    #     "10D.3vc4": [i for i in range(1001, 1015)],
    #     "10D.3vc5": [i for i in range(1016, 1046)],
    #     "10D.4vc4": [i for i in range(1101, 1115)],
    #     "10D.4vc5": [i for i in range(1116, 1146)],
    # }
    indexes = {
        "mut05": [i for i in range(0, 100)],
        "mut30": [i for i in range(100, 200)],
        "uniq1k": [i for i in range(200, 300)],
        "uniq5ksched": [i for i in range(300, 400)],
    }
    for key in indexes:
        compileGraphs(indexes[key], "CompiledXOR%s.png"%(key), key, "../resultDump", columns=3, get_all_electrode_EC=False)

if __name__== "__main__":
  main()