# Experiment to visualize the average occupancy of each 
# acceptor.

import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters
N = 50
M = 5
xdim = 1
ydim = 1
zdim = 0

#%% Initialize simulation object
kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, callback = 'callback_dwelltime')

kmc.I_0 = 100*kmc.kT
#%% Simulate
hops = 100000
kmc.simulate_discrete(hops = hops)

#%% Visualize
fig = kmc_dn_utils.visualize_dwelltime(kmc)
plt.show()

