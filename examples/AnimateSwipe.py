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
import dn_animation
import seaborn as sns
import copy

def openKmc(abs_file_path):
    kmc = kmc_dn.kmc_dn(10, 3, 1, 1, 0)
    kmc.loadSelf(abs_file_path)
    return kmc

def getSwipeResults(dn, bool_func, steps, hops):
    dn.swipe_results = []
    dn.electrodes[0][3] = dn.bool_voltage[bool_func[0][0]]
    dn.electrodes[1][3] = dn.bool_voltage[bool_func[0][1]]
    for i in range(1, len(bool_func)):
        print ("i is %d"%(i))
        from_voltage = [dn.electrodes[0][3], dn.electrodes[1][3]]
        to_voltage = [dn.bool_voltage[bool_func[i][0]], dn.bool_voltage[bool_func[i][1]]]
        for j in range(steps):
            dn.electrodes[0][3] = from_voltage[0] + (to_voltage[0]-from_voltage[0])*(j*1.0/steps)
            dn.electrodes[1][3] = from_voltage[1] + (to_voltage[1]-from_voltage[1])*(j*1.0/steps)
            dn.update_V()
            dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")
            dn.swipe_results.append((dn.electrodes.copy(), dn.current.copy(), dn.traffic.copy(), dn.time))
            print(j)
        dn.electrodes[0][3] = to_voltage[0]
        dn.electrodes[1][3] = to_voltage[1]
    print (dn.swipe_results)


def animateExample(index, useCalcs=False):
    xor = [(False, False, False), (True, False, True), (True, True, False), (False, True, True), (False, False, False)]
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    if not useCalcs:
        rel_path = "tests/trained/GeneticResultDumpVoltageGenetic%d.kmc"%(index)
    else:
        rel_path = "swipeResults/xor%d.kmc"%(index)
    
    abs_file_path = os.path.join(script_dir, rel_path)
    dn = openKmc(abs_file_path)
    print (dn.electrodes)
    if not hasattr(dn, "true_voltage"):
        dn.bool_voltage = {}
        dn.bool_voltage[True] = 75
        dn.bool_voltage[False] = 0
    else:
        dn.bool_voltage = {}
        dn.bool_voltage[True] = dn.true_voltage
        dn.bool_voltage[False] = 0
    #dn.electrodes[2][3] = 0
    if not hasattr(dn, "swipe_results"):
        getSwipeResults(dn, xor, 40, 5000000)
        rel_write_path = "swipeResults/xor%d.kmc"%(index)
        abs_file_path = os.path.join(script_dir, rel_write_path)
        dn.saveSelf(abs_file_path)
    writer = dn_animation.getWriter(20, "Swipe animation")
    dn_animation.trafficAnimation(dn, dn.swipe_results, writer, "swipe_animation%d.mp4"%(index), 40, 40)
  

def main():
    for index in [120, 121]:#range(7, 14):
        animateExample(index, False)
    
if __name__== "__main__":
  main()