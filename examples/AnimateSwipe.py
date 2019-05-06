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

def getSwipeResults(dn, bool_func, steps, hops, waits, shuffle):
    dn.swipe_results = []
    dn.electrodes[0][3] = bool_func[0][0]
    dn.electrodes[1][3] = bool_func[0][1]
    dn.update_V()
    
    for i in range(1, len(bool_func)):
        print ("i is %d"%(i))
        from_voltage = [dn.electrodes[0][3], dn.electrodes[1][3]]
        to_voltage = [bool_func[i][0], bool_func[i][1]]
        for j in range(waits):
            dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")
            dn.swipe_results.append((dn.electrodes.copy(), dn.current.copy(), dn.traffic.copy(), dn.time))
            if shuffle:
                dn.place_charges_random()
        for j in range(steps):
            dn.electrodes[0][3] = from_voltage[0] + (to_voltage[0]-from_voltage[0])*(j*1.0/steps)
            dn.electrodes[1][3] = from_voltage[1] + (to_voltage[1]-from_voltage[1])*(j*1.0/steps)
            dn.update_V()
            dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")
            dn.swipe_results.append((dn.electrodes.copy(), dn.current.copy(), dn.traffic.copy(), dn.time))
            if shuffle:
                dn.place_charges_random()
        dn.electrodes[0][3] = to_voltage[0]
        dn.electrodes[1][3] = to_voltage[1]
        dn.update_V()
    for j in range(waits):
        dn.go_simulation(hops = hops, record=True,  goSpecificFunction="wrapperSimulateRecord")
        dn.swipe_results.append((dn.electrodes.copy(), dn.current.copy(), dn.traffic.copy(), dn.time))
        if shuffle:
            dn.place_charges_random()

def animateExample(index, useCalcs=False, animation_index=None, shuffle=False):
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    if not useCalcs:
        rel_path = "../GeneticResultDumpVoltageGenetic%d.kmc"%(index)
    else:
        rel_path = "swipeResults/xor%d.kmc"%(index)
    
    abs_file_path = os.path.join(script_dir, rel_path)
    try:
        dn = openKmc(abs_file_path)
    except:
        return
    tests = dn.tests
    print (tests)
    values = []
    for test in tests:
        values.append(test[0])
    #dn.electrodes[2][3] = 0
    if not hasattr(dn, "swipe_results"):
        getSwipeResults(dn, values, 40, 5000000, 40, shuffle)
        rel_write_path = "swipeResults/xor%d.kmc"%(index)
        abs_file_path = os.path.join(script_dir, rel_write_path)
        dn.saveSelf(abs_file_path)
    writer = dn_animation.getWriter(10, "Swipe animation")
    if animation_index:
        animation_file = "swipe_animation%d_%d.mp4"%(index, animation_index)
    else:
        animation_file = "swipe_animation%d.mp4"%(index)
    dn_animation.trafficAnimation(dn, dn.swipe_results, writer, animation_file, 80, 0)
  

def main():
    for index in range(95, 120):
        animateExample(index, False)
    #animateExample(2, animation_index=1, shuffle=True)
    #animateExample(2, animation_index=2)
    #for index in range(10):
    #    animateExample(1, False, index)
if __name__== "__main__":
  main()