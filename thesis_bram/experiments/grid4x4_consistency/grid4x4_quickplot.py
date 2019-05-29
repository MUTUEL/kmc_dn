'''
Here I define a 4x4 grid of dopants on a 1x1 2D domain (two electrodes).
I will attempt to find suitable parameters to obtain non-linear/coulomb blockade
behaviour.
To this end I explore the following parameter space:
    kT = 1
    bias = [-100 kT, 100 kT]
    I_0 = [0.1kT, 100kT]
    ab = [0.01R, 5R]

'''
#%% Imports
import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt
import time
#%% System setup
xdim = 1
ydim = 1
acceptors = np.zeros((16, 3))
for i in range(4):
    for j in range(4):
        acceptors[4*i + j] = [(i+1)*xdim/5, (j+1)*ydim/5, 0]
donors = np.zeros((9, 3))
for i in range(3):
    for j in range(3):
        donors[3*i + j] = [i*xdim/5 + 1.5*xdim/5, j*ydim/5 + 1.5*ydim/5, 0]
electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 10]
electrodes[1] = [xdim, ydim/2, 0, 0]
kT = 1
I_0 = 1*kT 
ab_R = 0.1
VkT = 1000

#%% Initialize system
kmc = kmc_dn.kmc_dn(1, 5, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptors)
kmc.load_donors(donors)
kmc = kmc_dn.kmc_dn(16, 9, xdim, ydim, 0, electrodes=electrodes)
tic = time.time()
bias = np.linspace(-VkT*kT/kmc.e, VkT*kT/kmc.e, 100)
#%% Simulation loop

# Set constants
kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R
kmc.initialize(placement = False)
 # Simulate
current_sim = kmc_dn_utils.IV(kmc, 0, bias, prehops = 10000, hops = 10000) 
current = current_sim[:, 0] 
#%% Visualize
domain = kmc_dn_utils.visualize_basic(kmc)
plt.figure()
plt.plot(bias, current)
plt.show()

