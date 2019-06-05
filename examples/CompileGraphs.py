import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import kmc_dopant_networks as kmc_dn
import os
import sys

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
    
    tmp_swipe_results = dn.swipe_results[:40]
    tmp_swipe_results.extend(dn.swipe_results[80:120])
    tmp_swipe_results.extend(dn.swipe_results[160:200])
    tmp_swipe_results.extend(dn.swipe_results[240:280])
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
    ax.text(40, y, "(%d) Separation: %.3g"%(index, (true_min - false_max)))



def compileGraphs(indexes, fileOut):
    fig = plt.figure(figsize=(12, 20))
    gs = gridspec.GridSpec(7, 2)
    
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    i = 0
    for index in indexes:
        rel_path = "swipeResults/xor%d.kmc"%(index)
    
        abs_file_path = os.path.join(script_dir, rel_path)
        dn = openKmc(abs_file_path)
        drawGraph(dn, gs[i], fig, i+1)
        i+=1
    plt.savefig(fileOut)


def main():
    index5 = [i for i in range(8, 20)]
    index5.extend([6, 126])
    indexes = {
        # 5:index5,
        # 10:[i for i in range(46, 60)],
        # 20:[i for i in range(60, 74)],
        # 30:[i for i in range(74, 88)],
        # 45:[i for i in range(88, 102)],
        # 60:[i for i in range(102, 116)]
        1:[i for i in range(136, 149)]
    }
    for key in indexes:
        compileGraphs(indexes[key], "compiled_results%dDOP.png"%(key))

if __name__== "__main__":
  main()