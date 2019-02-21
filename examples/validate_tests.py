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


#%% Parameters

N = 30  # Number of acceptors
M = 4  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(2E5)
#res = 1  # Resolution of laplace grid

for i in range(200):
    # Define electrodes

    kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim)
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "tests/rnd/test"+str(i)+".kmc"
    abs_file_path = os.path.join(script_dir, rel_path)
    #%% Profile code
    kmc.loadSelf(abs_file_path)
    kmc.go_simulation(hops=hops)
    #kmc.current = kmc.current[0]
    diff_sum = 0.0
    curr_sum = 0.0
    print (kmc.current)
    print (kmc.expected_current)
    for i in range(len(kmc.current)):
        diff = math.fabs(kmc.current[i]-kmc.expected_current[i])
        diff_sum+=diff
        curr_sum+=math.fabs(kmc.current[i])
    
    print("Diff sum: %.2f, abs current sum: %.2f"%(diff_sum, curr_sum))