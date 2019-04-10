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


#%% Parameters

N = 30  # Number of acceptors
M = 4  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension
hops = int(1E6)
#res = 1  # Resolution of laplace grid

for i in range(200):
    # Define electrodes
    electrodes = np.zeros((8, 4))
    electrodes[0] = [0, ydim/4, 0, random.random()*320-160]
    electrodes[1] = [0, 3*ydim/4, 0, random.random()*320-160]
    electrodes[2] = [xdim, ydim/4, 0, random.random()*320-160]
    electrodes[3] = [xdim, 3*ydim/4, 0, random.random()*320-160]
    electrodes[4] = [xdim/4, 0, 0, random.random()*320-160]
    electrodes[5] = [3*xdim/4, 0, 0, random.random()*320-160]
    electrodes[6] = [xdim/4, ydim, 0, random.random()*320-160]
    electrodes[7] = [3*xdim/4, ydim, 0, 0]
    #%% Initialize simulation object

    kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes = electrodes)
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = "tests/rnd3/test"+str(i)+".kmc"
    abs_file_path = os.path.join(script_dir, rel_path)
    #%% Profile code
    kmc.python_simulation(hops=hops)
    print (kmc.time)
    print (kmc.electrode_occupation)
    print (kmc.current)
    kmc.saveSelf(abs_file_path)