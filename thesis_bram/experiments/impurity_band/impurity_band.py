import kmc_dopant_networks as kmc_dn
import kmc_dopant_networks_utils as kmc_dn_utils
import numpy as np
import matplotlib.pyplot as plt

#%% Parameters
N = 30
M = 3
xdim = 1
ydim = 1
zdim = 0

#%% Initialization
kmc = kmc_dn.kmc_dn(N, M, xdim, ydim, zdim) 

#%% Plot histogram of site energies (i.e. impurity band)
plt.hist(kmc.E_constant, bins = 50) 
plt.show()
