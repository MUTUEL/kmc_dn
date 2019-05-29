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
I_0 = np.logspace(-1, 2, 10)*kT
bias = np.linspace(-10*kT, 10*kT, 100)
ab_R = np.array([0.25])
current = np.zeros((len(I_0), len(ab_R), len(bias)))


#%% Initialize system
kmc = kmc_dn.kmc_dn(1, 0, xdim, ydim, 0, electrodes=electrodes)
kmc.load_acceptors(acceptors)
tic = time.time()
#%% Simulation loop
for i in range(len(I_0)):
    for j in range(len(ab_R)):

        # Set constants
        kmc.kT = kT
        kmc.I_0 = I_0[i]
        kmc.ab = ab_R[j]*kmc.R
        kmc.initialize(placement = False)

        # Simulate
        current_sim = kmc_dn_utils.IV(kmc, 0, bias, hops = 1000) 
        current[i, j] = current_sim[:, 0] 
print(f'{time.time() - tic}')
np.savez('current_map', current = current, I_0 = I_0, ab_R = ab_R, bias = bias)
#%% Visualize
domain = kmc_dn_utils.visualize_basic(kmc)
#plt.figure()
#plt.plot(bias, current[:, 0])
plt.show()

