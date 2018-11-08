'''
This file is a basic example of working with the kmc_dn class.
It illustrates initialization, simulation and visualization.
The setup here is a simple two electrode configuration on
a 1x1 2D domain with 10 acceptors.

@author: Bram de Wilde (b.dewilde-1@student.utwente.nl)
'''

import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters
N = 10  # Number of acceptors
M = 0  # Number of donors
xdim = 1  # Length along x dimension
ydim = 1  # Length along y dimension
zdim = 0  # Length along z dimension

# Define electrodes
electrodes = np.zeros((2, 4))
electrodes[0] = [0, ydim/2, 0, 10]  # Left electrode
electrodes[1] = [xdim, ydim/2, 0, -10] # Right electrode

#%% Initialize simulation object
kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim, electrodes = electrodes)

#%% Simulate
hops = int(1E4)
kmc.simulate_discrete(hops = hops)

#%% Visualize
domain = kmc_dn_utils.visualize_basic(kmc)
current_plot = kmc_dn_utils.visualize_current(kmc)
plt.show()
