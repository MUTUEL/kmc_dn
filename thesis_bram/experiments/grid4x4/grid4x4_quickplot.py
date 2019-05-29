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
electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 10]
electrodes[1] = [xdim, ydim/2, 0, 0]
kT = 1
bias = np.linspace(-10*kT, 10*kT, 100)
I_0 = 1000
ab_R = 0.1


#%% Initialize system
kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptors)
tic = time.time()
#%% Simulation loop

# Set constants
kmc.kT = kT
kmc.I_0 = I_0
kmc.ab = ab_R*kmc.R
kmc.initialize(placement = False)
 # Simulate
current_sim = kmc_dn_utils.IV(kmc, 0, bias, interval = 1000) 
current = current_sim[:, 0] 
#%% Visualize

plt.figure()
plt.plot(bias, current)
plt.show()

