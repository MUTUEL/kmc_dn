'''
Here I will make some IV curves to check for the basic shape
on various random layouts with 30 acceptors/3 donors. 
Also using in principle values around:
    kT = 1
    bias = +/-100kT
    I_0 = kT
    a_b = 0.1R
'''
#%% Imports
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import time

#%% sim parameters
prehops = 1000
hops = 100000
#%% Load random layouts
acceptor_layouts = np.load('acceptor_layouts.npy')
donor_layouts = np.load('donor_layouts.npy')
layout = np.arange(0, 10)
kT = 1
I_0 = [1*kT]
bias = np.linspace(-100*kT, 100*kT, 100)
ab_R = [0.25]
current = np.zeros((len(layout), len(I_0), len(ab_R), len(bias)))

for k in range(len(layout)):
    #%% System setup
    xdim = 1
    ydim = 1
    acceptors = acceptor_layouts[layout[k]]
    donors = donor_layouts[layout[k]]
    electrodes = np.zeros((2, 4))
    electrodes[0] = [0, ydim/2, 0, 10]
    electrodes[1] = [xdim, ydim/2, 0, 0]
    
    #%% Initialize system
    kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes, callback = 'none')
    kmc.load_acceptors(acceptors)
    kmc.load_donors(donors)
    #%% Simulation loop
    for i in range(len(I_0)):
        for j in range(len(ab_R)):

            # Set constants
            kmc.kT = kT
            kmc.I_0 = I_0[i]
            kmc.ab = ab_R[j]*kmc.R
            kmc.initialize(dopant_placement = False)

            # Simulate
            current_sim = kmc_dn_utils.IV(kmc, 0, bias, prehops = prehops, hops = hops)
            current[k, i, j] = current_sim[:, 1]


#np.savez('data', current = current, I_0 = I_0, ab_R = ab_R, bias = bias)
#%% Visualize
domain = kmc_dn_utils.visualize_basic(kmc)
plt.figure()
for i in range(current.shape[0]):
    plt.plot(bias, current[i, 0, 0])
plt.show()
